"""Tool registration and execution."""

import logging
from dataclasses import dataclass
from typing import Any, Callable

from src.tools.bash import BashResult, BashTool

logger = logging.getLogger(__name__)


@dataclass
class ToolDefinition:
    """Definition of a registered tool."""

    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[..., Any]
    requires_approval: bool = False


class ToolRegistry:
    """Registry for agent tools."""

    def __init__(
        self,
        approval_callback: Callable[[str, str], bool] | None = None,
    ) -> None:
        """
        Initialize the tool registry.

        Args:
            approval_callback: Function to call for tool approval (tool_name, details) -> bool
        """
        self._tools: dict[str, ToolDefinition] = {}
        self._approval_callback = approval_callback
        self._bash_tool = BashTool(
            approval_callback=lambda cmd: self._request_approval("bash", cmd)
        )

        # Register built-in tools
        self._register_builtin_tools()

    def _request_approval(self, tool_name: str, details: str) -> bool:
        """Request approval for tool execution."""
        if self._approval_callback:
            return self._approval_callback(tool_name, details)
        return False

    def _register_builtin_tools(self) -> None:
        """Register built-in tools."""
        # Bash tool
        self.register(
            name="bash",
            description="Execute bash commands in a sandboxed environment",
            parameters={
                "command": {
                    "type": "string",
                    "description": "The bash command to execute",
                    "required": True,
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds",
                    "required": False,
                    "default": 30,
                },
            },
            handler=self._execute_bash,
            requires_approval=True,
        )

        # File read tool (safe)
        self.register(
            name="read_file",
            description="Read contents of a file",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file to read",
                    "required": True,
                },
                "max_lines": {
                    "type": "integer",
                    "description": "Maximum number of lines to read",
                    "required": False,
                    "default": 100,
                },
            },
            handler=self._read_file,
            requires_approval=False,
        )

        # List directory tool (safe)
        self.register(
            name="list_dir",
            description="List contents of a directory",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the directory",
                    "required": True,
                },
            },
            handler=self._list_dir,
            requires_approval=False,
        )

    def _execute_bash(self, command: str, timeout: int = 30) -> str:
        """Execute bash command handler."""
        result = self._bash_tool.execute(command, timeout=timeout)

        if result.blocked:
            return f"[BLOCKED] {result.block_reason}"

        if result.timed_out:
            return f"[TIMEOUT] Command timed out after {timeout} seconds"

        output = result.stdout
        if result.stderr:
            output += f"\n[STDERR]\n{result.stderr}"
        if result.return_code != 0:
            output += f"\n[EXIT CODE: {result.return_code}]"

        return output or "[No output]"

    def _read_file(self, path: str, max_lines: int = 100) -> str:
        """Read file contents handler."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            if len(lines) > max_lines:
                content = "".join(lines[:max_lines])
                content += f"\n... ({len(lines) - max_lines} more lines)"
            else:
                content = "".join(lines)

            return content

        except FileNotFoundError:
            return f"[ERROR] File not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def _list_dir(self, path: str) -> str:
        """List directory contents handler."""
        import os

        try:
            entries = os.listdir(path)
            if not entries:
                return "[Empty directory]"

            result_lines = []
            for entry in sorted(entries):
                full_path = os.path.join(path, entry)
                if os.path.isdir(full_path):
                    result_lines.append(f"  {entry}/")
                else:
                    size = os.path.getsize(full_path)
                    result_lines.append(f"  {entry} ({size} bytes)")

            return "\n".join(result_lines)

        except FileNotFoundError:
            return f"[ERROR] Directory not found: {path}"
        except PermissionError:
            return f"[ERROR] Permission denied: {path}"
        except Exception as e:
            return f"[ERROR] {str(e)}"

    def register(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Any],
        requires_approval: bool = False,
    ) -> None:
        """
        Register a new tool.

        Args:
            name: Unique tool name
            description: Tool description
            parameters: Parameter schema
            handler: Function to handle tool execution
            requires_approval: Whether tool requires user approval
        """
        if name in self._tools:
            logger.warning(f"Overwriting existing tool: {name}")

        self._tools[name] = ToolDefinition(
            name=name,
            description=description,
            parameters=parameters,
            handler=handler,
            requires_approval=requires_approval,
        )
        logger.debug(f"Registered tool: {name}")

    def get_tool(self, name: str) -> ToolDefinition | None:
        """Get a tool definition by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[ToolDefinition]:
        """List all registered tools."""
        return list(self._tools.values())

    def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> str:
        """
        Execute a tool by name.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool

        Returns:
            Tool output as string
        """
        tool = self._tools.get(tool_name)

        if not tool:
            return f"[ERROR] Unknown tool: {tool_name}"

        # Check required parameters
        for param_name, param_spec in tool.parameters.items():
            if param_spec.get("required", False) and param_name not in arguments:
                return f"[ERROR] Missing required parameter: {param_name}"

        # Apply defaults
        final_args = {}
        for param_name, param_spec in tool.parameters.items():
            if param_name in arguments:
                final_args[param_name] = arguments[param_name]
            elif "default" in param_spec:
                final_args[param_name] = param_spec["default"]

        try:
            result = tool.handler(**final_args)
            return str(result)

        except Exception as e:
            logger.error(f"Tool execution failed: {tool_name} - {e}")
            return f"[ERROR] Tool execution failed: {str(e)}"

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """
        Get tool schemas in a format suitable for LLM function calling.

        Returns:
            List of tool schemas
        """
        schemas = []
        for tool in self._tools.values():
            # Convert to JSON Schema format
            properties = {}
            required = []

            for param_name, param_spec in tool.parameters.items():
                properties[param_name] = {
                    "type": param_spec.get("type", "string"),
                    "description": param_spec.get("description", ""),
                }
                if param_spec.get("required", False):
                    required.append(param_name)

            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            })

        return schemas
