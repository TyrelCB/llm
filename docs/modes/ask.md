# Ask Mode

Routing bias: **retrieve** (prioritizes KB retrieval).

```
User Query
   |
   v
Route Query (mode bias = retrieve)
   |-- !command? -----------------> Execute Tools (explicit) -> END
   |
   v
Retrieve Docs -> Grade Docs
   |-- relevant ---------> Generate Response -> END
   |-- none/low ----------> Rewrite Query -> Retrieve (loop, max N)
   |                        |-- retries left -> ...
   |                        |-- exhausted ----> Fallback (external)
   |                                         -> Update KB? -> END
```

Notes:
- Ask mode strongly prefers retrieval; tools are only used via explicit `!command`.
