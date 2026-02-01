# Chat Mode

Routing bias: **balanced** (LLM router decides between tool / retrieve / generate).

Explicit `!command` always bypasses routing and executes directly.

```
User Query
   |
   v
Route Query (LLM router)
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
- Retrieval may rewrite the query up to `settings.max_rewrite_attempts` before fallback.
- KB updates only occur on external fallback responses.
