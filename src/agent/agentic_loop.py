"""Agentic control loop with compact state and persistent logs."""

from __future__ import annotations

import json
import os
import re
import shlex
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

from langchain_core.messages import HumanMessage, SystemMessage

from config.settings import settings
from src.knowledge.retriever import Retriever
from src.llm.selector import ProviderSelector
from src.tools.registry import ToolRegistry


StepApprovalCallback = Callable[[dict[str, Any]], bool]
AskUserCallback = Callable[[str], str]
BashApprovalCallback = Callable[[str], bool]
StepResultCallback = Callable[[dict[str, Any], dict[str, Any]], None]


@dataclass
class AgenticResult:
    """Result from an agentic run."""

    response: str
    tool_results: list[dict[str, Any]]
    steps_executed: int
    stopped_early: bool


class AgenticLoop:
    """Runs an agentic loop with a small working set and persistent state."""

    def __init__(
        self,
        state_dir: Path | None = None,
        selector: ProviderSelector | None = None,
        retriever: Retriever | None = None,
        tool_registry: ToolRegistry | None = None,
        bash_approval_callback: BashApprovalCallback | None = None,
        ask_user_callback: AskUserCallback | None = None,
    ) -> None:
        self._state_dir = state_dir or settings.agentic_state_dir
        self._selector = selector or ProviderSelector()
        self._retriever = retriever or Retriever()
        self._tool_registry = tool_registry or ToolRegistry(
            approval_callback=bash_approval_callback,
        )
        self._ask_user_callback = ask_user_callback

        self._logs_dir = self._state_dir / "logs"
        self._scratch_dir = self._state_dir / "scratch"
        self._state_path = self._state_dir / "state.json"
        self._steps_log = self._logs_dir / "steps.jsonl"
        self._scratch_output = self._scratch_dir / "tool_output.txt"

        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=True)
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._scratch_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        task: str,
        approval_mode: str = "step",
        step_approval_callback: StepApprovalCallback | None = None,
        step_result_callback: StepResultCallback | None = None,
        max_steps: int | None = None,
    ) -> AgenticResult:
        state = self._init_state(task)
        max_steps = max_steps or settings.agentic_max_steps
        tool_results: list[dict[str, Any]] = []
        stopped_early = False

        for step_index in range(1, max_steps + 1):
            step_start = time.time()
            state["step"] = step_index
            state["goal_step"] = state.get("goal_step", 0) + 1
            action = self._bootstrap_action(state)
            if action is None:
                facts = self._retrieve_facts(state)
                prompt = self._build_prompt(state, facts)
                action = self._request_action(prompt, state)

            if action.get("status") == "final":
                final_text = self._safe_text(action.get("final", ""))
                state["last_action"] = "final"
                state["last_result_summary"] = self._trim_text(
                    final_text,
                    settings.agentic_last_result_max_chars,
                )
                self._save_state(state)
                self._log_step(state, action, {"output": final_text})
                continue_loop, response = self._complete_goal(state, final_text)
                if continue_loop:
                    continue
                return AgenticResult(
                    response=response,
                    tool_results=tool_results,
                    steps_executed=step_index,
                    stopped_early=False,
                )

            action_signature = self._action_signature(action)
            repeat_count = state.get("repeat_count", 0)
            if action_signature and action_signature == state.get("last_action_signature", ""):
                repeat_count += 1
            else:
                repeat_count = 0
            state["repeat_count"] = repeat_count

            if repeat_count >= settings.agentic_repeat_limit:
                action = self._repeat_breaker_action(state)
                action_signature = self._action_signature(action)
                state["repeat_count"] = 0

            if approval_mode == "step" and step_approval_callback:
                approved = step_approval_callback(action)
                if not approved:
                    stopped_early = True
                    state["status"] = "stopped"
                    self._save_state(state)
                    self._log_step(state, action, {"output": "Step approval denied"})
                    break

            tool_start = time.time()
            result = self._execute_action(action)
            result["tool_duration_ms"] = int((time.time() - tool_start) * 1000)
            tool_results.append(result)
            summary = self._summarize_output(result.get("output", ""))
            result["summary"] = summary
            self._maybe_store_tts_path(state, action, result)
            self._update_no_match_count(state, action, result)
            if result.get("tool") == "ask_user" and not result.get("success", False):
                state["status"] = "needs_user_input"
                state["last_action"] = self._format_action(action)
                state["last_result_summary"] = summary
                self._save_state(state)
                self._log_step(state, action, result)
                if step_result_callback:
                    step_result_callback(action, result)
                return AgenticResult(
                    response=result.get("output", ""),
                    tool_results=tool_results,
                    steps_executed=step_index,
                    stopped_early=True,
                )
            result["step_duration_ms"] = int((time.time() - step_start) * 1000)
            state["last_action"] = self._format_action(action)
            state["last_result_summary"] = summary
            state["last_action_signature"] = action_signature
            self._apply_state_update(state, action.get("state_update", {}))
            self._save_state(state)
            self._log_step(state, action, result)
            if step_result_callback:
                step_result_callback(action, result)

            no_match_final = self._maybe_no_match_finalize(state, action, result)
            if no_match_final:
                continue_loop, response = self._complete_goal(state, no_match_final)
                if continue_loop:
                    continue
                return AgenticResult(
                    response=response,
                    tool_results=tool_results,
                    steps_executed=step_index,
                    stopped_early=False,
                )

            auto_final = self._maybe_auto_finalize(state, action, result)
            if auto_final:
                continue_loop, response = self._complete_goal(state, auto_final)
                if continue_loop:
                    continue
                return AgenticResult(
                    response=response,
                    tool_results=tool_results,
                    steps_executed=step_index,
                    stopped_early=False,
                )

        if stopped_early:
            response = "Agentic run stopped before completion."
        else:
            response = "Agentic run reached the maximum step limit."

        return AgenticResult(
            response=response,
            tool_results=tool_results,
            steps_executed=state.get("step", max_steps),
            stopped_early=stopped_early,
        )

    def _update_no_match_count(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        action_spec = action.get("action") or {}
        if action_spec.get("tool") != "search":
            state["no_match_count"] = 0
            return
        output = self._safe_text(result.get("output", "")).strip()
        if output == "[No matches]":
            state["no_match_count"] = state.get("no_match_count", 0) + 1
        else:
            state["no_match_count"] = 0

    def _maybe_no_match_finalize(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        action_spec = action.get("action") or {}
        if action_spec.get("tool") != "search":
            return None
        output = self._safe_text(result.get("output", "")).strip()
        if output != "[No matches]":
            return None
        goal = state.get("goal", "").lower()
        threshold = settings.agentic_no_match_limit
        if any(term in goal for term in ("wrong", "issue", "problem", "bug", "review")):
            threshold = 1
        if state.get("no_match_count", 0) < threshold:
            return None

        if not any(term in goal for term in ("wrong", "issue", "problem", "bug", "review")):
            return None

        return (
            "Quick scan found no obvious TODO/FIXME/BUG markers in the repo. "
            "That doesn't guarantee correctness. For a real review, I recommend:\n"
            "- Read key modules under `src/` (agent graph, tools, settings)\n"
            "- Run `ruff check .` and `mypy src`\n"
            "- Run `pytest` for behavioral coverage\n"
            "If you want, tell me a target area (e.g., agent loop, tools) and I'll inspect it."
        )

    def _init_state(self, task: str) -> dict[str, Any]:
        run_id = uuid.uuid4().hex[:8]
        goals = self._parse_goals(task)
        if not goals:
            goals = [task.strip()]
        last_tts_path = self._load_last_tts_path()
        state = {
            "run_id": run_id,
            "goal": goals[0].strip(),
            "goals": goals,
            "goal_index": 0,
            "goal_summaries": [],
            "goal_step": 0,
            "last_tts_path": last_tts_path,
            "constraints": [],
            "progress": [],
            "open_questions": [],
            "last_action": "",
            "last_result_summary": "",
            "last_action_signature": "",
            "repeat_count": 0,
            "no_match_count": 0,
            "plan": [],
            "step": 0,
            "status": "in_progress",
            "updated_at": time.time(),
        }
        self._save_state(state)
        return state

    def _load_last_tts_path(self) -> str:
        if not self._state_path.exists():
            return ""
        try:
            raw = self._state_path.read_text()
            data = json.loads(raw)
        except Exception:
            return ""
        path = data.get("last_tts_path")
        if isinstance(path, str) and path.strip():
            return path.strip()
        return ""

    def _reset_goal_state(self, state: dict[str, Any]) -> None:
        state["constraints"] = []
        state["progress"] = []
        state["open_questions"] = []
        state["last_action"] = ""
        state["last_result_summary"] = ""
        state["last_action_signature"] = ""
        state["repeat_count"] = 0
        state["no_match_count"] = 0
        state["plan"] = []
        state["goal_step"] = 0

    def _format_goal_summary(self, state: dict[str, Any], summary: str) -> str:
        goals = state.get("goals", [])
        index = state.get("goal_index", 0)
        goal_text = state.get("goal", "")
        if len(goals) <= 1:
            return summary
        header = f"Goal {index + 1}: {goal_text}"
        return f"{header}\n{summary}"

    def _complete_goal(self, state: dict[str, Any], summary: str) -> tuple[bool, str]:
        summaries = state.setdefault("goal_summaries", [])
        summaries.append(self._format_goal_summary(state, summary))
        goals = state.get("goals", [])
        index = state.get("goal_index", 0)
        if index + 1 < len(goals):
            state["goal_index"] = index + 1
            state["goal"] = goals[state["goal_index"]].strip()
            self._reset_goal_state(state)
            self._save_state(state)
            return True, ""

        state["status"] = "completed"
        self._save_state(state)
        return False, "\n\n".join(summaries)

    def _save_state(self, state: dict[str, Any]) -> None:
        state["updated_at"] = time.time()
        self._state_path.write_text(json.dumps(state, indent=2))

    def _retrieve_facts(self, state: dict[str, Any]) -> list[dict[str, str]]:
        query_parts = [state.get("goal", "")]
        query_parts.extend(state.get("open_questions", []))
        query = " ".join(p for p in query_parts if p).strip()
        if not query:
            return []

        documents = self._retriever.retrieve(query=query, k=settings.agentic_facts_k)
        facts: list[dict[str, str]] = []

        for doc in documents[: settings.agentic_facts_k]:
            snippet = " ".join(doc.page_content.split())
            snippet = self._trim_text(snippet, settings.agentic_fact_max_chars)
            source = str(doc.metadata.get("source", "unknown"))
            facts.append({"source": source, "snippet": snippet})

        return facts

    def _build_prompt(self, state: dict[str, Any], facts: list[dict[str, str]]) -> str:
        state_card = self._render_state_card(state)
        facts_text = self._render_facts(facts)
        tools_text = self._render_tools()
        schema_text = self._render_schema()

        return (
            f"Task:\n{state.get('goal', '')}\n\n"
            f"State:\n{state_card}\n\n"
            f"Facts:\n{facts_text}\n\n"
            f"Tools:\n{tools_text}\n\n"
            f"Output JSON schema:\n{schema_text}"
        )

    def _request_action(self, prompt: str, state: dict[str, Any]) -> dict[str, Any]:
        system_prompt = (
            "You are an agent. Respond with JSON only. "
            "Choose exactly one action per step. "
            "Never invent tool results; use tools when evidence is required. "
            "If system state or files are involved, prefer bash/search/read_file first. "
            "For system health checks, prefer bash: uptime, df -h, free -h, ps. "
            "For codebase overviews, read README.md and AGENTS.md (if present) first. "
            "The tool name must be one of: bash, search, read_file, write_file, list_dir, ask_user. "
            "Tool arguments must be JSON objects, not tool-call strings. "
            "If enough information is available, set status to 'final' and fill final. "
            "Never include markdown or explanations outside JSON."
        )
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ]
        max_attempts = max(0, settings.agentic_action_retries)
        for attempt in range(max_attempts + 1):
            result = self._selector.generate(messages, force_local=True)
            action, error = self._parse_action(result.content)
            if action is not None:
                return action
            messages.append(
                HumanMessage(
                    content=(
                        "Your previous response was invalid. "
                        f"Error: {error}. "
                        "Reply with valid JSON only, matching the schema."
                    ),
                )
            )
            if attempt >= max_attempts:
                break
        return {
            "status": "final",
            "final": self._fallback_final(state),
        }

    def _parse_action(self, content: str) -> tuple[dict[str, Any] | None, str]:
        text = content.strip()
        try:
            action = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    action = json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    return None, "Invalid JSON"
            return None, "Invalid JSON"

        valid, error = self._validate_action(action)
        if not valid:
            return None, error
        return action, ""

    def _fallback_final(self, state: dict[str, Any]) -> str:
        goal = state.get("goal", "").lower()
        if any(term in goal for term in ("wrong", "issue", "problem", "bug", "review")):
            return (
                "I couldn't complete a reliable automated review. "
                "Quick scan didn't surface obvious error strings. "
                "For a real review, I recommend:\n"
                "- Read key modules under `src/` (agent graph, tools, settings)\n"
                "- Run `ruff check .` and `mypy src`\n"
                "- Run `pytest` for behavioral coverage\n"
                "Tell me a target area and I'll inspect it."
            )
        if any(term in goal for term in ("tts", "text to speech", "speech")):
            last_tts = state.get("last_tts_path", "")
            if last_tts:
                return f"TTS audio generated at: {last_tts}"
            return "TTS request was not completed. Please provide the text to synthesize."
        return "Unable to parse action JSON after retries. Please try again."

    @staticmethod
    def _parse_goals(task: str) -> list[str]:
        if not task:
            return []
        separators = r"\b(?:and then|then|after that|next)\b[,:]?\s+"
        parts = re.split(separators, task, flags=re.IGNORECASE)
        goals: list[str] = []
        for part in parts:
            chunk = part.strip()
            if not chunk:
                continue
            comma_split = re.split(
                r"[;,]\s+(?=(?:then|and then|after that|next|also|please|tell|check|do|find|review|summarize|analyze|list|show|explain|describe)\b)",
                chunk,
                flags=re.IGNORECASE,
            )
            for item in comma_split:
                cleaned = item.strip(" ,.;")
                if cleaned:
                    goals.append(cleaned)
        return goals

    def _validate_action(self, action: dict[str, Any]) -> tuple[bool, str]:
        if not isinstance(action, dict):
            return False, "Action must be a JSON object"

        status = action.get("status")
        if status not in ("continue", "final"):
            return False, "status must be 'continue' or 'final'"

        if status == "final":
            return True, ""

        action_spec = action.get("action")
        if not isinstance(action_spec, dict):
            return False, "action must be an object"

        tool = action_spec.get("tool")
        allowed_tools = self._allowed_tools()
        if tool not in allowed_tools:
            allowed = ", ".join(sorted(allowed_tools))
            return False, f"tool must be one of: {allowed}"

        args = action_spec.get("args", {})
        if args is not None and not isinstance(args, dict):
            return False, "args must be an object"

        if not self._validate_tool_args(tool, args or {}):
            return False, "Invalid tool arguments"

        return True, ""

    def _validate_tool_args(self, tool: str, args: dict[str, Any]) -> bool:
        if tool == "bash":
            return self._is_valid_arg_string(args.get("command"))
        if tool == "search":
            if not self._is_valid_arg_string(args.get("query")):
                return False
            path = args.get("path")
            return path is None or self._is_valid_arg_string(path, allow_braces=False)
        if tool == "read_file":
            return self._is_valid_arg_string(args.get("path"), allow_braces=False)
        if tool == "list_dir":
            return self._is_valid_arg_string(args.get("path"), allow_braces=False)
        if tool == "write_file":
            if not self._is_valid_arg_string(args.get("path"), allow_braces=False):
                return False
            return isinstance(args.get("content"), str)
        if tool == "ask_user":
            return isinstance(args.get("question"), str) and bool(args.get("question"))
        return True

    @staticmethod
    def _is_valid_arg_string(value: Any, allow_braces: bool = True) -> bool:
        if not isinstance(value, str) or not value.strip():
            return False
        if not allow_braces and ("{" in value or "}" in value):
            return False
        return True

    def _allowed_tools(self) -> set[str]:
        return {tool.name for tool in self._tool_registry.list_tools()} | {"ask_user"}

    def _maybe_store_tts_path(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        action_spec = action.get("action") or {}
        tool = action_spec.get("tool")
        output = self._safe_text(result.get("output", ""))
        if tool == "tts":
            marker = "TTS generated:"
            if marker in output:
                path = output.split(marker, 1)[1].strip()
                if path:
                    clean_path = path.split(" (", 1)[0].strip()
                    state["last_tts_path"] = clean_path
            return
        if tool == "ask_user":
            question = self._safe_text(action_spec.get("args", {}).get("question", ""))
            if "audio file path" in question.lower():
                candidate = self._extract_audio_path(output)
                if candidate:
                    state["last_tts_path"] = candidate
                    state["open_questions"] = [
                        item
                        for item in state.get("open_questions", [])
                        if "audio file path" not in item.lower()
                    ]

    @staticmethod
    def _extract_quoted_text(text: str) -> str | None:
        match = re.search(r"\"([^\"]+)\"|'([^']+)'", text)
        if not match:
            return None
        return match.group(1) or match.group(2)

    def _extract_audio_path(self, text: str) -> str:
        cleaned = text.strip().strip("\"'").strip()
        if not cleaned or cleaned.lower() in {"none", "n/a", "no"}:
            return ""
        path_candidate = Path(cleaned).expanduser()
        if path_candidate.is_absolute() and path_candidate.exists():
            return str(path_candidate)
        match = re.search(r"(/[^\"'\s]+?\.(?:mp3|wav|m4a|flac|ogg))", text)
        if match:
            matched_path = Path(match.group(1)).expanduser()
            if matched_path.exists():
                return str(matched_path)
            return ""
        if not path_candidate.is_absolute():
            relative = (settings.project_root / path_candidate).resolve()
            if relative.exists():
                return str(relative)
        return ""

    def _audio_play_command(self, path: str) -> str:
        quoted = shlex.quote(path)
        return (
            "play_cmd=\"\"; "
            f"if command -v ffplay >/dev/null 2>&1; then play_cmd=\"ffplay -nodisp -autoexit -nostdin -loglevel error {quoted}\"; "
            f"elif command -v mpv >/dev/null 2>&1; then play_cmd=\"mpv --no-video --quiet {quoted}\"; "
            f"elif command -v mpg123 >/dev/null 2>&1; then play_cmd=\"mpg123 -q {quoted}\"; "
            f"elif command -v paplay >/dev/null 2>&1; then play_cmd=\"paplay {quoted}\"; "
            f"elif command -v ffmpeg >/dev/null 2>&1 && command -v aplay >/dev/null 2>&1; then play_cmd=\"ffmpeg -loglevel error -i {quoted} -f wav - | aplay -q\"; "
            f"elif command -v aplay >/dev/null 2>&1; then play_cmd=\"aplay {quoted}\"; "
            "fi; "
            "if [ -z \"$play_cmd\" ]; then "
            "echo \"[ERROR] No audio playback tool found (ffplay/mpv/mpg123/paplay/ffmpeg+aplay/aplay)\"; "
            "exit 1; "
            "fi; "
            "if command -v nohup >/dev/null 2>&1; then "
            "nohup sh -c \"$play_cmd\" >/dev/null 2>&1 & "
            "else "
            "sh -c \"$play_cmd\" >/dev/null 2>&1 & "
            "fi; "
            "echo \"[OK] Audio playback started\""
        )

    def _bootstrap_action(self, state: dict[str, Any]) -> dict[str, Any] | None:
        if state.get("goal_step") != 1:
            return None
        goal = state.get("goal", "").lower()
        if any(term in goal for term in ("play the generated file", "play audio", "play the audio", "play the file")):
            last_tts = state.get("last_tts_path", "")
            requested_path = self._extract_audio_path(state.get("goal", ""))
            play_path = requested_path or last_tts
            if play_path and Path(play_path).exists():
                state["last_tts_path"] = play_path
                return {
                    "status": "continue",
                    "thought": "Play the last generated TTS file.",
                    "action": {
                        "tool": "bash",
                        "args": {"command": self._audio_play_command(play_path)},
                    },
                    "state_update": {"progress_add": ["Play last TTS output"]},
                }
            if play_path:
                return {
                    "status": "continue",
                    "thought": "Need a valid audio file path before playback.",
                    "action": {
                        "tool": "ask_user",
                        "args": {
                            "question": (
                                f"I couldn't find '{play_path}'. "
                                "What is the audio file path to play?"
                            )
                        },
                    },
                    "state_update": {"open_questions_add": ["Need audio file path to play"]},
                }
            return {
                "status": "continue",
                "thought": "Need the path to the audio file before playback.",
                "action": {"tool": "ask_user", "args": {"question": "What is the audio file path to play?"}},
                "state_update": {"open_questions_add": ["Need audio file path to play"]},
            }

        if "tts" in goal or "text to speech" in goal or "speech" in goal:
            text = self._extract_quoted_text(state.get("goal", ""))
            if text:
                mode = "custom_voice"
                if "voice clone" in goal or "clone" in goal:
                    mode = "voice_clone"
                elif "voice design" in goal or "design" in goal:
                    mode = "voice_design"
                return {
                    "status": "continue",
                    "thought": "Generate TTS audio.",
                    "action": {
                        "tool": "tts",
                        "args": {
                            "mode": mode,
                            "text": text,
                            "language": "auto",
                        },
                    },
                    "state_update": {"progress_add": ["Generated TTS audio"]},
                }

        if any(term in goal for term in ("pytest", "test")):
            return {
                "status": "continue",
                "thought": "Run pytest to check test status.",
                "action": {
                    "tool": "bash",
                    "args": {"command": "pytest"},
                },
                "state_update": {"progress_add": ["Run pytest"]},
            }
        if any(term in goal for term in ("wrong", "issue", "problem", "bug", "review")):
            return {
                "status": "continue",
                "thought": "Scan for obvious code markers like TODO/FIXME/BUG.",
                "action": {
                    "tool": "search",
                    "args": {
                        "query": "TODO|FIXME|BUG|HACK|XXX",
                        "path": str(settings.project_root),
                        "max_results": 50,
                    },
                },
                "state_update": {"progress_add": ["Scan repo for TODO/FIXME/BUG/HACK markers"]},
            }
        if not any(term in goal for term in ("codebase", "repository", "repo")):
            return None

        candidates = ["README.md", "AGENTS.md", "CLAUDE.md"]
        for name in candidates:
            path = settings.project_root / name
            if path.exists():
                return {
                    "status": "continue",
                    "thought": f"Read {name} for codebase overview.",
                    "action": {"tool": "read_file", "args": {"path": str(path), "max_lines": 200}},
                    "state_update": {"progress_add": [f"Read {name}"]},
                }

        return None

    def _execute_action(self, action: dict[str, Any]) -> dict[str, Any]:
        action_spec = action.get("action") or {}
        tool_name = action_spec.get("tool")
        tool_args = action_spec.get("args", {})
        if not isinstance(tool_args, dict):
            tool_args = {}

        if tool_name == "ask_user":
            question = self._safe_text(tool_args.get("question", ""))
            if self._ask_user_callback:
                answer = self._ask_user_callback(question)
                return {
                    "tool": tool_name,
                    "args": tool_args,
                    "output": answer,
                    "success": True,
                }
            return {
                "tool": tool_name,
                "args": tool_args,
                "output": f"CLARIFY: {question}",
                "success": False,
                "error": "ask_user unavailable",
            }

        if not tool_name:
            return {
                "tool": "",
                "args": tool_args,
                "output": "[ERROR] Missing tool name",
                "success": False,
                "error": "missing tool",
            }

        output = self._tool_registry.execute(tool_name, tool_args)
        success = not str(output).startswith("[ERROR]") and "[BLOCKED]" not in str(output)
        return {
            "tool": tool_name,
            "args": tool_args,
            "output": output,
            "success": success,
        }

    def _apply_state_update(self, state: dict[str, Any], update: dict[str, Any]) -> None:
        for key, field in (
            ("progress_add", "progress"),
            ("open_questions_add", "open_questions"),
            ("constraints_add", "constraints"),
            ("plan_add", "plan"),
        ):
            additions = update.get(key, [])
            if isinstance(additions, str):
                additions = [additions]
            if additions:
                state[field].extend(self._trim_list(additions))

        removals = update.get("open_questions_remove", [])
        if isinstance(removals, str):
            removals = [removals]
        if removals:
            state["open_questions"] = [
                item for item in state.get("open_questions", [])
                if item not in removals
            ]

    def _render_state_card(self, state: dict[str, Any]) -> str:
        trimmed = {
            "goal": self._trim_text(state.get("goal", ""), 240),
            "constraints": self._trim_list(state.get("constraints", [])),
            "progress": self._trim_list(state.get("progress", [])),
            "open_questions": self._trim_list(state.get("open_questions", [])),
            "last_action": self._trim_text(state.get("last_action", ""), 160),
            "last_result_summary": self._trim_text(
                state.get("last_result_summary", ""),
                settings.agentic_last_result_max_chars,
            ),
            "last_tts_path": self._trim_text(state.get("last_tts_path", ""), 200),
        }
        text = json.dumps(trimmed, indent=2)
        if len(text) <= settings.agentic_state_max_chars:
            return text

        trimmed["progress"] = trimmed["progress"][:3]
        trimmed["open_questions"] = trimmed["open_questions"][:3]
        trimmed["constraints"] = trimmed["constraints"][:3]
        text = json.dumps(trimmed, indent=2)
        if len(text) <= settings.agentic_state_max_chars:
            return text

        trimmed["progress"] = []
        trimmed["open_questions"] = []
        trimmed["constraints"] = []
        return json.dumps(trimmed, indent=2)

    def _render_facts(self, facts: list[dict[str, str]]) -> str:
        if not facts:
            return "- (none)"
        lines = []
        for fact in facts:
            source = fact.get("source", "unknown")
            snippet = fact.get("snippet", "")
            lines.append(f"- {snippet} (source: {source})")
        return "\n".join(lines)

    def _render_tools(self) -> str:
        tool_specs = [
            "bash(command, timeout=30) - run a shell command (system health: uptime, df -h, free -h, ps -eo pid,pcpu,pmem,cmd --sort=-pmem | head)",
            "search(query, path='.', max_results=50) - ripgrep search (glob like *.py lists files)",
            "read_file(path, max_lines=200) - read a file",
            "write_file(path, content, mode='overwrite', create_dirs=true) - write file",
            "list_dir(path) - list directory contents",
            "tts(mode, text, language='auto', speaker=None, instruct=None, ref_audio_path=None, ref_text=None) - text to speech",
            "ask_user(question) - request missing info",
        ]
        return "\n".join(f"{idx + 1}. {tool}" for idx, tool in enumerate(tool_specs))

    def _render_schema(self) -> str:
        schema = {
            "status": "continue|final",
            "thought": "short sentence",
            "action": {"tool": "tool_name", "args": {}},
            "state_update": {
                "progress_add": [],
                "open_questions_add": [],
                "open_questions_remove": [],
                "constraints_add": [],
                "plan_add": [],
            },
            "final": "",
        }
        return json.dumps(schema, indent=2)

    def _format_action(self, action: dict[str, Any]) -> str:
        spec = action.get("action") or {}
        tool = spec.get("tool", "")
        args = spec.get("args", {})
        if not tool:
            return "unknown"
        return f"{tool}({args})"

    def _action_signature(self, action: dict[str, Any]) -> str:
        spec = action.get("action") or {}
        tool = spec.get("tool", "")
        args = spec.get("args", {})
        try:
            args_text = json.dumps(args, sort_keys=True)
        except TypeError:
            args_text = str(args)
        return f"{tool}:{args_text}" if tool else ""

    def _repeat_breaker_action(self, state: dict[str, Any]) -> dict[str, Any]:
        question = (
            "I keep repeating the same step. What should I check next or "
            "which tool should I use?"
        )
        return {
            "status": "continue",
            "thought": "Detected a loop; ask for guidance.",
            "action": {"tool": "ask_user", "args": {"question": question}},
            "state_update": {
                "open_questions_add": [question],
            },
        }

    def _maybe_auto_finalize(
        self,
        state: dict[str, Any],
        action: dict[str, Any],
        result: dict[str, Any],
    ) -> str | None:
        goal = state.get("goal", "").lower()
        if "health" in goal:
            action_spec = action.get("action") or {}
            if action_spec.get("tool") != "bash":
                return None
            command = str((action_spec.get("args") or {}).get("command", "")).lower()
            if "uptime" not in command or "df -h" not in command or "free -h" not in command:
                return None
            output = result.get("output", "")
            if not output:
                return None
            return self._summarize_system_health(output)

        if any(term in goal for term in ("pytest", "test")):
            action_spec = action.get("action") or {}
            if action_spec.get("tool") != "bash":
                return None
            command = str((action_spec.get("args") or {}).get("command", "")).lower()
            if "pytest" not in command:
                return None
            output = result.get("output", "")
            if not output:
                return None
            return self._summarize_pytest(output)

        if any(term in goal for term in ("codebase", "repository", "repo")):
            action_spec = action.get("action") or {}
            if action_spec.get("tool") != "read_file":
                return None
            path = str((action_spec.get("args") or {}).get("path", "")).lower()
            if not path:
                return None
            if not any(name in path for name in ("readme.md", "agents.md", "claude.md")):
                return None
            output = result.get("output", "")
            if not output:
                return None
            return self._summarize_codebase(output)

        return None

    def _summarize_codebase(self, output: str) -> str:
        title = self._extract_heading(output) or "Local codebase"
        features = self._extract_section_bullets(output, "Features")
        structure = (
            self._extract_section_bullets(output, "Project Structure & Module Organization")
            or self._extract_section_bullets(output, "Project Structure")
        )

        lines = [f"Codebase summary: {title}"]
        if features:
            lines.append("Key features:")
            lines.extend(f"- {item}" for item in features[:6])
        if structure:
            lines.append("Key directories:")
            lines.extend(f"- {item}" for item in structure[:6])

        if len(lines) == 1:
            excerpt = self._extract_intro_lines(output)
            if excerpt:
                lines.append(excerpt)

        return "\n".join(lines)

    @staticmethod
    def _extract_heading(text: str) -> str | None:
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("# "):
                return line[2:].strip()
        return None

    @staticmethod
    def _extract_section_bullets(text: str, header: str) -> list[str]:
        lines = text.splitlines()
        start = None
        header_lower = header.lower()
        for idx, line in enumerate(lines):
            if line.strip().lower() in (f"## {header_lower}", f"# {header_lower}"):
                start = idx + 1
                break
        if start is None:
            return []

        bullets: list[str] = []
        for line in lines[start:]:
            if line.strip().startswith("#"):
                break
            stripped = line.strip()
            if stripped.startswith(("-", "*")):
                bullets.append(stripped.lstrip("-*").strip())
        return bullets

    @staticmethod
    def _extract_intro_lines(text: str) -> str:
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            lines.append(stripped)
            if len(lines) >= 2:
                break
        return " ".join(lines)

    def _summarize_system_health(self, output: str) -> str:
        summary_lines = []
        warnings = []

        load_match = re.search(
            r"load average: ([0-9.]+),\s*([0-9.]+),\s*([0-9.]+)",
            output,
        )
        if load_match:
            load_1 = float(load_match.group(1))
            load_5 = float(load_match.group(2))
            load_15 = float(load_match.group(3))
            cpus = os.cpu_count() or 1
            summary_lines.append(
                f"Load average: {load_1:.2f}, {load_5:.2f}, {load_15:.2f} (CPUs: {cpus})"
            )
            if load_1 > cpus:
                warnings.append("Load average is above CPU count; system may be under heavy load.")

        for line in output.splitlines():
            if line.strip().endswith(" /") and "%" in line:
                parts = line.split()
                try:
                    use_percent = int(parts[4].rstrip("%"))
                    summary_lines.append(
                        f"Disk usage /: {parts[2]} used of {parts[1]} ({use_percent}%)"
                    )
                    if use_percent >= 90:
                        warnings.append("Root filesystem usage is above 90%.")
                except (IndexError, ValueError):
                    continue
                break

        mem_match = re.search(
            r"^Mem:\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(\S+)",
            output,
            re.MULTILINE,
        )
        if mem_match:
            total = self._parse_size(mem_match.group(1))
            available = self._parse_size(mem_match.group(6))
            if total and available is not None:
                avail_pct = (available / total) * 100
                summary_lines.append(f"Memory available: {avail_pct:.1f}%")
                if avail_pct < 10:
                    warnings.append("Available memory is below 10%.")

        ps_header_index = None
        lines = output.splitlines()
        for idx, line in enumerate(lines):
            if line.strip().startswith("PID"):
                ps_header_index = idx
                break
        if ps_header_index is not None and ps_header_index + 1 < len(lines):
            top_line = lines[ps_header_index + 1].strip()
            if top_line:
                summary_lines.append(f"Top memory process: {top_line}")

        if not summary_lines:
            return "System health check completed, but no metrics were parsed."

        report = ["System health summary:"] + summary_lines
        if warnings:
            report.append("Warnings:")
            report.extend(f"- {warning}" for warning in warnings)
        return "\n".join(report)

    def _summarize_pytest(self, output: str) -> str:
        lines = output.splitlines()
        summary = []
        warnings = []
        errors = []

        for line in lines:
            if "collected" in line and "items" in line:
                summary.append(line.strip())
            if "PytestConfigWarning" in line:
                warnings.append(line.strip())
            if line.strip().startswith("ERROR:"):
                errors.append(line.strip())

        if not summary:
            for line in lines:
                if "test session starts" in line:
                    summary.append(line.strip())
                    break

        if not summary:
            summary.append("Pytest run completed.")

        report = ["Pytest summary:"] + summary
        if errors:
            report.append("Errors:")
            report.extend(f"- {error}" for error in errors[:3])
        if warnings:
            report.append("Warnings:")
            report.extend(f"- {warning}" for warning in warnings[:3])

        if any("collected 0 items" in line for line in summary):
            report.append("No tests were collected; check testpaths or add tests.")

        return "\n".join(report)

    @staticmethod
    def _parse_size(value: str) -> float | None:
        try:
            number = float(value[:-1])
            unit = value[-1].upper()
        except ValueError:
            return None
        scale = {
            "K": 1024,
            "M": 1024 ** 2,
            "G": 1024 ** 3,
            "T": 1024 ** 4,
        }.get(unit)
        if scale is None:
            return None
        return number * scale

    def _summarize_output(self, output: str) -> str:
        raw = self._safe_text(output)
        snippet = self._trim_text(raw, settings.agentic_tool_output_max_chars)
        summary = self._trim_text(snippet, settings.agentic_last_result_max_chars)
        self._scratch_output.write_text(snippet)
        return summary

    def _log_step(self, state: dict[str, Any], action: dict[str, Any], result: dict[str, Any]) -> None:
        summary = result.get("summary") or self._summarize_output(result.get("output", ""))
        entry = {
            "timestamp": time.time(),
            "run_id": state.get("run_id"),
            "step": state.get("step"),
            "action": action,
            "result_summary": summary,
            "tool_duration_ms": result.get("tool_duration_ms", 0),
            "step_duration_ms": result.get("step_duration_ms", 0),
        }
        with self._steps_log.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    @staticmethod
    def _trim_text(text: str, max_chars: int) -> str:
        text = text.strip()
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."

    @staticmethod
    def _trim_list(items: Iterable[str], max_items: int = 6, max_chars: int = 200) -> list[str]:
        trimmed: list[str] = []
        for item in list(items)[:max_items]:
            trimmed.append(AgenticLoop._trim_text(str(item), max_chars))
        return trimmed

    @staticmethod
    def _safe_text(value: Any) -> str:
        if value is None:
            return ""
        return str(value)
