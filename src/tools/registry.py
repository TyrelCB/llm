"""Tool registration and execution."""

import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import httpx

from config.settings import settings
from src.tools.bash import BashTool

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

        # TTS tool (service-first, fallback to local)
        self.register(
            name="tts",
            description="Generate speech audio from text via TTS service",
            parameters={
                "mode": {
                    "type": "string",
                    "description": "custom_voice|voice_design|voice_clone",
                    "required": False,
                    "default": "custom_voice",
                },
                "text": {
                    "type": "string",
                    "description": "Text to synthesize",
                    "required": True,
                },
                "language": {
                    "type": "string",
                    "description": "Language hint (e.g., auto, en, zh)",
                    "required": False,
                    "default": "auto",
                },
                "speaker": {
                    "type": "string",
                    "description": "Speaker preset (custom_voice)",
                    "required": False,
                },
                "instruct": {
                    "type": "string",
                    "description": "Voice instruction (custom_voice/voice_design)",
                    "required": False,
                },
                "ref_audio_path": {
                    "type": "string",
                    "description": "Reference audio path (voice_clone)",
                    "required": False,
                },
                "ref_text": {
                    "type": "string",
                    "description": "Reference transcript (voice_clone)",
                    "required": False,
                },
            },
            handler=self._tts,
            requires_approval=False,
        )

        # Search tool (safe)
        self.register(
            name="search",
            description="Search files using ripgrep (supports glob patterns)",
            parameters={
                "query": {
                    "type": "string",
                    "description": "Search query for ripgrep",
                    "required": True,
                },
                "path": {
                    "type": "string",
                    "description": "Path to search within",
                    "required": False,
                    "default": ".",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Maximum number of matches to return",
                    "required": False,
                    "default": 50,
                },
            },
            handler=self._search,
            requires_approval=False,
        )

        # File write tool (enabled by default)
        self.register(
            name="write_file",
            description="Write contents to a file",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Path to the file to write",
                    "required": True,
                },
                "content": {
                    "type": "string",
                    "description": "Content to write",
                    "required": True,
                },
                "mode": {
                    "type": "string",
                    "description": "Write mode: overwrite or append",
                    "required": False,
                    "default": "overwrite",
                },
                "create_dirs": {
                    "type": "boolean",
                    "description": "Create parent directories if missing",
                    "required": False,
                    "default": True,
                },
            },
            handler=self._write_file,
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

    def _tts(
        self,
        text: str,
        mode: str = "custom_voice",
        language: str = "auto",
        speaker: str | None = None,
        instruct: str | None = None,
        ref_audio_path: str | None = None,
        ref_text: str | None = None,
    ) -> str:
        """Generate speech audio via TTS service with local fallback."""
        payload = {
            "mode": mode,
            "text": text,
            "language": language,
            "speaker": speaker,
            "instruct": instruct,
            "ref_audio_path": ref_audio_path,
            "ref_text": ref_text,
        }
        try:
            response = httpx.post(
                f"{settings.tts_service_url}/tts",
                json=payload,
                timeout=300,
            )
            response.raise_for_status()
            data = response.json()
            return (
                f"[OK] TTS generated: {data.get('path')} "
                f"(mode={data.get('mode')}, sr={data.get('sample_rate')}, "
                f"duration_ms={data.get('duration_ms')})"
            )
        except Exception as exc:
            fallback = self._tts_local(payload)
            if fallback:
                return fallback
            return f"[ERROR] TTS service failed: {exc}"

    def _tts_local(self, payload: dict[str, str | None]) -> str | None:
        """Fallback to in-process Qwen3-TTS if available."""
        try:
            import numpy as np
            import soundfile as sf
            from qwen_tts import Qwen3TTSModel
            import torch
        except Exception:
            return None

        def resolve_device() -> str:
            if settings.tts_device != "auto":
                return settings.tts_device
            if torch.cuda.is_available():
                return "cuda"
            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
            return "cpu"

        def resolve_dtype(device: str) -> str:
            if device == "cpu" and settings.tts_dtype in ("float16", "bfloat16"):
                return "float32"
            return settings.tts_dtype

        device = resolve_device()
        dtype = resolve_dtype(device)

        mode = payload.get("mode") or "custom_voice"
        if mode == "custom_voice":
            model_id = settings.tts_model_custom
        elif mode == "voice_design":
            model_id = settings.tts_model_design
        else:
            model_id = settings.tts_model_clone

        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype,
        )

        text = payload.get("text") or ""
        language = payload.get("language") or "auto"
        speaker = payload.get("speaker")
        instruct = payload.get("instruct")
        ref_audio_path = payload.get("ref_audio_path")
        ref_text = payload.get("ref_text")

        if mode == "custom_voice":
            wavs, sr = model.generate_custom_voice(
                text=text,
                language=language,
                speaker=speaker,
                instruct=instruct,
            )
        elif mode == "voice_design":
            wavs, sr = model.generate_voice_design(
                text=text,
                language=language,
                instruct=instruct,
            )
        else:
            if not ref_audio_path or not ref_text:
                return "[ERROR] voice_clone requires ref_audio_path and ref_text"
            wavs, sr = model.generate_voice_clone(
                text=text,
                language=language,
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )

        audio = wavs[0] if isinstance(wavs, (list, tuple)) else wavs
        output_dir = Path(settings.tts_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_path = output_dir / f"tts_{os.getpid()}_{os.urandom(3).hex()}"
        wav_path = base_path.with_suffix(".wav")
        sf.write(wav_path, audio, sr)

        if settings.tts_output_format == "wav":
            return f"[OK] TTS generated: {wav_path} (mode={mode}, sr={sr})"

        if shutil.which("ffmpeg") is None:
            return f"[ERROR] ffmpeg not available for mp3 output; wrote {wav_path}"

        mp3_path = base_path.with_suffix(".mp3")
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", str(wav_path), str(mp3_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return f"[ERROR] ffmpeg failed: {result.stderr.strip()}"

        return f"[OK] TTS generated: {mp3_path} (mode={mode}, sr={sr})"

    def _search(self, query: str, path: str = ".", max_results: int = 50) -> str:
        """Search files using ripgrep or a fallback scan."""
        search_path = Path(path).expanduser()
        if not search_path.is_absolute():
            search_path = (settings.project_root / search_path).resolve()

        if not search_path.exists():
            return f"[ERROR] Search path not found: {search_path}"

        is_glob = any(ch in query for ch in ("*", "?", "["))
        if shutil.which("rg"):
            exclude_globs = ["!**/.git/*", "!**/.agent/*"]
            if is_glob and " " not in query and "/" not in query:
                cmd = [
                    "rg",
                    "--files",
                    "--glob",
                    query,
                ]
                for glob in exclude_globs:
                    cmd.extend(["--glob", glob])
                cmd.append(str(search_path))
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode not in (0, 1):
                    return f"[ERROR] rg failed: {result.stderr.strip()}"
                lines = result.stdout.splitlines()
                return "\n".join(lines[:max_results]) or "[No matches]"

            cmd = [
                "rg",
                "--line-number",
                "--no-heading",
                "--max-count",
                str(max_results),
            ]
            for glob in exclude_globs:
                cmd.extend(["--glob", glob])
            cmd.extend([query, str(search_path)])
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode not in (0, 1):
                return f"[ERROR] rg failed: {result.stderr.strip()}"
            lines = result.stdout.splitlines()
            return "\n".join(lines[:max_results]) or "[No matches]"

        matches: list[str] = []
        for root, dirs, files in os.walk(search_path):
            dirs[:] = [d for d in dirs if d not in {".git", ".agent"}]
            for name in files:
                file_path = Path(root) / name
                try:
                    if is_glob:
                        from fnmatch import fnmatch
                        if fnmatch(name, query):
                            matches.append(str(file_path))
                            if len(matches) >= max_results:
                                return "\n".join(matches)
                        continue
                    with file_path.open("r", encoding="utf-8") as handle:
                        for idx, line in enumerate(handle, 1):
                            if query in line:
                                matches.append(f"{file_path}:{idx}:{line.rstrip()}")
                                if len(matches) >= max_results:
                                    return "\n".join(matches)
                except (UnicodeDecodeError, PermissionError):
                    continue

        return "\n".join(matches) or "[No matches]"

    def _write_file(
        self,
        path: str,
        content: str,
        mode: str = "overwrite",
        create_dirs: bool = True,
    ) -> str:
        """Write file contents with path safety checks."""
        target = Path(path).expanduser()
        if not target.is_absolute():
            target = (settings.project_root / target).resolve()

        root = settings.project_root.resolve()
        if root not in target.parents and target != root:
            return f"[ERROR] Path outside project root: {target}"

        if mode not in ("overwrite", "append"):
            return f"[ERROR] Invalid mode: {mode}. Use 'overwrite' or 'append'."

        if create_dirs:
            target.parent.mkdir(parents=True, exist_ok=True)

        write_mode = "a" if mode == "append" else "w"
        try:
            with target.open(write_mode, encoding="utf-8") as handle:
                handle.write(content)
            return f"[OK] Wrote {len(content)} chars to {target}"
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
