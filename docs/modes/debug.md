# Debug Mode

Routing bias: **balanced** (same flow as chat), but with verbose routing/tool logs enabled.

```
User Query
   |
   v
Route Query (LLM router + verbose logs)
   |-- !command? -----------------> Execute Tools (explicit) -> END
   |
   |-- tool ----------------------> Prepare Tools -> Execute Tools -> Generate Response -> END
   |
   |-- retrieve ------------------> Retrieve Docs -> Grade Docs
   |                                   |-- relevant ---------> Generate Response -> END
   |                                   |-- none/low ----------> Rewrite Query -> Retrieve (loop, max N)
   |                                   |                        |-- retries left -> ...
   |                                   |                        |-- exhausted ----> Fallback (external)
   |                                   |                                         -> Update KB? -> END
   |
   |-- generate ------------------> Generate Response -> END
```

Notes:
- Same routing as Chat mode, but emits `[DEBUG]` logs for routing and tools.
