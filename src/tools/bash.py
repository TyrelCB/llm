"""Sandboxed bash execution tool with allow/deny pattern matching."""

import logging
import os
import re
import subprocess
from dataclasses import dataclass
from typing import Callable

from config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class BashResult:
    """Result from bash command execution."""

    stdout: str
    stderr: str
    return_code: int
    command: str
    timed_out: bool = False
    blocked: bool = False
    block_reason: str = ""


class BashTool:
    """
    Bash command execution with allow/deny pattern matching.

    Default mode is PERMISSIVE: allow all commands except those matching deny patterns.
    Deny patterns can require approval or be completely blocked.
    """

    # Patterns that are ALWAYS blocked (catastrophic/irreversible)
    BLOCKED_PATTERNS = [
        r"rm\s+-[rf]*\s+/\s*$",          # rm -rf /
        r"rm\s+-[rf]*\s+/\*",             # rm -rf /*
        r"rm\s+-[rf]*\s+~\s*$",           # rm -rf ~
        r"rm\s+-[rf]*\s+~/\*",            # rm -rf ~/*
        r"mkfs\s+",                        # Format filesystem
        r"dd\s+if=/dev/(zero|random|urandom)\s+of=/dev/",  # Disk wipe
        r">\s*/dev/sd[a-z]",              # Redirect to disk
        r"chmod\s+-R\s+777\s+/",          # World-writable root
        r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
    ]

    # Patterns that require APPROVAL before running (dangerous but sometimes needed)
    APPROVAL_PATTERNS = [
        r"rm\s+-[rf]+",                   # rm with force/recursive (but not root)
        r"chmod\s+",                       # Permission changes
        r"chown\s+",                       # Ownership changes
        r"curl\s+.*\|\s*(ba)?sh",         # curl pipe to shell
        r"wget\s+.*\|\s*(ba)?sh",         # wget pipe to shell
        r"eval\s+",                        # eval commands
        r"kill\s+-9",                      # Force kill
        r"pkill\s+",                       # Process kill
        r"killall\s+",                     # Kill all by name
        r"shutdown",                       # System shutdown
        r"reboot",                         # System reboot
        r"systemctl\s+(stop|restart|disable|mask|unmask)",  # Service control
        r"service\s+\w+\s+(stop|restart)", # Service control (old style)
    ]

    # Session-approved patterns (commands user has approved for this session)
    _session_approved: set[str] = set()

    def __init__(
        self,
        approval_callback: Callable[[str], bool] | None = None,
        working_dir: str | None = None,
        permissive: bool = True,
    ) -> None:
        """
        Initialize the bash tool.

        Args:
            approval_callback: Function to call for approval of dangerous commands
            working_dir: Working directory for command execution
            permissive: If True (default), allow all except denied. If False, require explicit allow.
        """
        self._approval_callback = approval_callback
        self._working_dir = working_dir or os.getcwd()
        self._permissive = permissive

    @classmethod
    def approve_for_session(cls, pattern: str) -> None:
        """Add a pattern to session-approved list."""
        cls._session_approved.add(pattern)
        logger.info(f"Approved pattern for session: {pattern}")

    @classmethod
    def clear_session_approvals(cls) -> None:
        """Clear all session approvals."""
        cls._session_approved.clear()

    def _is_blocked(self, command: str) -> tuple[bool, str]:
        """Check if command matches blocked patterns (always denied)."""
        for pattern in self.BLOCKED_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Blocked: matches dangerous pattern"
        return False, ""

    def _needs_approval(self, command: str) -> tuple[bool, str]:
        """Check if command needs approval before running."""
        # Check if already approved for session
        for approved in self._session_approved:
            try:
                if re.search(approved, command, re.IGNORECASE):
                    return False, ""
            except re.error:
                if approved in command:
                    return False, ""

        # Check approval patterns
        for pattern in self.APPROVAL_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Requires approval: potentially dangerous operation"
        return False, ""

    def _sanitize_output(self, output: str, max_length: int = 10000) -> str:
        """Sanitize and truncate command output."""
        if len(output) > max_length:
            output = output[:max_length] + f"\n... (truncated, {len(output)} total chars)"
        return output

    def validate_command(self, command: str) -> tuple[bool, str, bool]:
        """
        Validate a command before execution.

        Args:
            command: The command to validate

        Returns:
            Tuple of (is_valid, reason, requires_approval)
        """
        # Always check blocked patterns first
        is_blocked, reason = self._is_blocked(command)
        if is_blocked:
            return False, reason, False

        # Check if needs approval
        needs_approval, reason = self._needs_approval(command)
        if needs_approval:
            return True, reason, True

        # In permissive mode, allow everything else
        return True, "Allowed", False

    def execute(
        self,
        command: str,
        timeout: int | None = None,
        skip_approval: bool = False,
    ) -> BashResult:
        """
        Execute a bash command with safety checks.

        Args:
            command: The command to execute
            timeout: Timeout in seconds (defaults to settings)
            skip_approval: Skip approval check (use with caution)

        Returns:
            BashResult with execution results
        """
        timeout = timeout or settings.bash_timeout

        # Validate command
        is_valid, reason, requires_approval = self.validate_command(command)

        if not is_valid:
            logger.warning(f"Blocked command: {command} - {reason}")
            return BashResult(
                stdout="",
                stderr=reason,
                return_code=-1,
                command=command,
                blocked=True,
                block_reason=reason,
            )

        # Check approval if needed
        if requires_approval and not skip_approval:
            if self._approval_callback:
                approved = self._approval_callback(command)
                if not approved:
                    return BashResult(
                        stdout="",
                        stderr="User denied approval",
                        return_code=-1,
                        command=command,
                        blocked=True,
                        block_reason="User denied approval",
                    )
                # Approve similar commands for session
                base_cmd = command.split()[0] if command.split() else command
                self.approve_for_session(f"^{re.escape(base_cmd)}\\s")
            else:
                # No callback - block
                return BashResult(
                    stdout="",
                    stderr=f"{reason}. Use interactive mode or approve the command.",
                    return_code=-1,
                    command=command,
                    blocked=True,
                    block_reason=reason,
                )

        # Execute the command
        logger.info(f"Executing command: {command}")

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self._working_dir,
                env={**os.environ, "LANG": "C.UTF-8"},
            )

            return BashResult(
                stdout=self._sanitize_output(result.stdout),
                stderr=self._sanitize_output(result.stderr),
                return_code=result.returncode,
                command=command,
            )

        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out after {timeout}s: {command}")
            return BashResult(
                stdout="",
                stderr=f"Command timed out after {timeout} seconds",
                return_code=-1,
                command=command,
                timed_out=True,
            )

        except Exception as e:
            logger.error(f"Command execution failed: {e}")
            return BashResult(
                stdout="",
                stderr=str(e),
                return_code=-1,
                command=command,
            )
