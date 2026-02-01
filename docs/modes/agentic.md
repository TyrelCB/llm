# Agentic Mode

Routing: **bypasses the graph** and runs the AgenticLoop (auto mode, no prompts).

```
User Task
   |
   v
AgenticLoop
   |
   v
Init State
   |
   +--> Bootstrap Action? (health/test/codebase/tts shortcuts)
   |
   +--> Retrieve Facts (KB)
   |
   +--> Build Prompt (state + facts + tools + schema)
   |
   +--> LLM JSON Action
          |
          v
       Validate Action
          |
          v
       Execute Tool (bash/search/read_file/write_file/list_dir/tts)
          |
          v
       Summarize Output + Update State
          |
          v
       Auto-finalize? (health/test/codebase summaries)
          |
          v
       Loop until final or max steps
```

Notes:
- Agentic mode is always **auto** (no step approvals or ask_user prompts).
- Tools are executed via the ToolRegistry with safety checks.
