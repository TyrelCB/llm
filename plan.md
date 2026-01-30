# Agentic Loop Upgrade Plan

## Goals
- Add a dedicated agentic mode with a persistent loop, compact state, and hard caps.
- Keep prompts tiny and structured (JSON-only actions, one tool per step).
- Support both per-step approval and auto-run, with safe defaults.

## Phases
- [x] Phase 0: Define the agentic schema (state card, action JSON, caps) and prompt template.
- [x] Phase 1: Implement the agentic controller loop with `.agent/` persistence and step logs.
- [x] Phase 2: Expand the tool set (search, write_file, ask_user) and add output summarization.
- [x] Phase 3: Wire CLI/API mode integration, approvals (step vs auto), and docs updates.
- [ ] Phase 4: Add grounding improvements (evidence packets + deterministic retrieval first) and tests.
- [x] Post-Phase: Show per-step tool results and improve agentic tool guidance.
- [x] Post-Phase: Add auto-summary for common system health checks.
- [x] Post-Phase: Improve search tool with glob support and repo exclusions.
- [x] Post-Phase: Add deterministic codebase summaries from README/AGENTS.
- [x] Post-Phase: Retry invalid JSON and run agentic queries synchronously in CLI.
- [x] Post-Phase: Add no-match guard for repeated empty searches.
- [x] Post-Phase: Add codebase bootstrap and strict tool validation.
- [x] Post-Phase: Add multi-goal sequential runs for chained prompts.
- [x] Post-Phase: Expand multi-goal parsing to commas and validate tool args.
- [x] Post-Phase: Bootstrap review goals with TODO/FIXME/BUG scan and goal-step tracking.
- [x] Post-Phase: Add pytest bootstrap and summary for test-related goals.
- [x] Post-Phase: Add TTS tool + external service integration (CPU-first).
- [x] Post-Phase: Add TTS bootstrap and playback helper with last file tracking.
- [x] Post-Phase: Persist last TTS path across runs and improve playback fallbacks.
- [x] Post-Phase: Make playback non-blocking so the CLI returns after starting audio.
- [x] Post-Phase: Add per-step timing (tool + step duration) for agentic runs.

## Model Switching Fix (Jan 2026)
- [x] Sync runtime model changes to settings for new selectors and tools.
- [x] Reset cached model-dependent helpers on model switch.
- [x] Update docs and version for the behavior change.

## Project Root Defaults (Jan 2026)
- [x] Add CLI project-root override with cwd default.
- [x] Rebase derived paths when project root changes.
- [x] Update docs and version for the behavior change.

## Code Mode File Application (Jan 2026)
- [x] Require FILE blocks in code mode for file edits.
- [x] Auto-apply FILE blocks in the CLI.
- [x] Update docs for code mode file writes.
