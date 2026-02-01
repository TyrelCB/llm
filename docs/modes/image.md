# Image Mode

Routing bias: **image** (goes directly to image generation).

```
User Query
   |
   v
Route Query (mode bias = image)
   |
   v
Generate Image
   |-- Enhance Prompt (LLM)
   |-- Invoke Image Tool (ComfyUI / A1111)
   |
   v
Response (prompt + saved path)
```
