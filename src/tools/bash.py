"""Sandboxed bash execution tool."""

import logging
import os
import re
import shlex
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
    """Sandboxed bash command execution with safety checks."""

    # Commands that are always blocked
    BLOCKED_COMMANDS = {
        "rm -rf /",
        "rm -rf /*",
        "rm -rf ~",
        "rm -rf ~/*",
        "mkfs",
        "dd if=/dev/zero",
        "dd if=/dev/random",
        "> /dev/sda",
        "chmod -R 777 /",
        "chown -R",
        ":(){:|:&};:",  # Fork bomb
    }

    # Command patterns that require approval
    DANGEROUS_PATTERNS = [
        r"rm\s+-[rf]+\s+",  # rm with force/recursive
        r"sudo\s+",  # sudo commands
        r"chmod\s+",  # permission changes
        r"chown\s+",  # ownership changes
        r">\s*/",  # redirect to root
        r"curl\s+.*\|\s*(ba)?sh",  # curl pipe to shell
        r"wget\s+.*\|\s*(ba)?sh",  # wget pipe to shell
        r"eval\s+",  # eval commands
        r"exec\s+",  # exec commands
        r"kill\s+-9",  # force kill
        r"pkill\s+",  # process kill
        r"shutdown",  # system shutdown
        r"reboot",  # system reboot
        r"systemctl\s+(stop|disable|mask)",  # service control
        r"iptables",  # firewall rules
        r"dd\s+",  # disk operations
    ]

    # Safe command prefixes
    SAFE_PREFIXES = [
        "ls", "pwd", "echo", "cat", "head", "tail", "grep", "find",
        "wc", "sort", "uniq", "cut", "tr", "date", "whoami", "hostname",
        "env", "printenv", "which", "type", "file", "stat", "du", "df",
        "ps", "top", "free", "uptime", "uname", "id", "groups",
        "git status", "git log", "git diff", "git branch", "git show",
        "python --version", "pip list", "pip show",
        "node --version", "npm list",
        "docker ps", "docker images",
    ]

    def __init__(
        self,
        approval_callback: Callable[[str], bool] | None = None,
        working_dir: str | None = None,
    ) -> None:
        """
        Initialize the bash tool.

        Args:
            approval_callback: Function to call for approval of dangerous commands
            working_dir: Working directory for command execution
        """
        self._approval_callback = approval_callback
        self._working_dir = working_dir or os.getcwd()

    def _is_blocked(self, command: str) -> tuple[bool, str]:
        """Check if command is in the blocked list."""
        command_lower = command.lower().strip()

        for blocked in self.BLOCKED_COMMANDS:
            if blocked in command_lower:
                return True, f"Command contains blocked pattern: {blocked}"

        return False, ""

    def _is_dangerous(self, command: str) -> tuple[bool, str]:
        """Check if command matches dangerous patterns."""
        for pattern in self.DANGEROUS_PATTERNS:
            if re.search(pattern, command, re.IGNORECASE):
                return True, f"Command matches dangerous pattern: {pattern}"

        return False, ""

    def _is_safe(self, command: str) -> bool:
        """Check if command is in the safe list."""
        command_stripped = command.strip()
        for prefix in self.SAFE_PREFIXES:
            if command_stripped.startswith(prefix):
                return True
        return False

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
        # Check blocked commands
        is_blocked, reason = self._is_blocked(command)
        if is_blocked:
            return False, reason, False

        # Check if it's a safe command
        if self._is_safe(command):
            return True, "Safe command", False

        # Check if it's dangerous and needs approval
        is_dangerous, reason = self._is_dangerous(command)
        if is_dangerous:
            if settings.bash_require_approval:
                return True, reason, True
            else:
                logger.warning(f"Executing dangerous command without approval: {command}")
                return True, reason, False

        # Default: allow with approval if configured
        if settings.bash_require_approval:
            return True, "Unknown command type", True

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
                        stderr="Command requires approval - denied by user",
                        return_code=-1,
                        command=command,
                        blocked=True,
                        block_reason="User denied approval",
                    )
            else:
                return BashResult(
                    stdout="",
                    stderr=f"Command requires approval: {reason}",
                    return_code=-1,
                    command=command,
                    blocked=True,
                    block_reason=f"Requires approval: {reason}",
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

    def execute_safe(self, command: str) -> BashResult:
        """
        Execute a command only if it's in the safe list.

        Args:
            command: The command to execute

        Returns:
            BashResult with execution results
        """
        if not self._is_safe(command):
            return BashResult(
                stdout="",
                stderr="Command not in safe list",
                return_code=-1,
                command=command,
                blocked=True,
                block_reason="Not in safe command list",
            )

        return self.execute(command, skip_approval=True)
