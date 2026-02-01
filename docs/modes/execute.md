# Execute Mode

Routing bias: **tool** (prefers command/tool execution).

```
User Query
   |
   v
Route Query (mode bias = tool)
   |-- !command? -----------------> Execute Tools (explicit) -> END
   |
   v
Prepare Tools (LLM command generation)
   |
   v
Execute Tools
   |
   v
Generate Response (interpret tool output)
   |
   v
END
```

Notes:
- Explicit `!command` returns raw output and skips LLM interpretation.
