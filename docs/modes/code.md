# Code Mode

Routing bias: **generate** (direct response; no tool/retrieve routing).

```
User Query
   |
   v
Route Query (mode bias = generate)
   |
   v
Generate Response (code-focused system prompt)
   |
   v
END
```

Notes:
- In the CLI, if the response includes `FILE:` blocks, they are written directly to disk.
