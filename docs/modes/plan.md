# Plan Mode

Routing bias: **generate** (always generates a plan-style response; no tool/retrieve routing).

```
User Query
   |
   v
Route Query (mode bias = generate)
   |
   v
Generate Response (plan-style system prompt)
   |
   v
END
```

Related CLI command: `/plan` (separate pipeline)

```
/plan <task>
   |
   v
Generate Plan (LLM)
   |
   v
User Approval (y/a/n)
   |-- y (step) --> Execute $commands step-by-step
   |-- a ---------> Execute all $commands
   |-- n ---------> Cancel
   |
   v
Execution Report + Optional Analysis
```
