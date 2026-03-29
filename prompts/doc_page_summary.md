---
name: Page Summarizer
description: Generates 1-2 sentence summaries for each page of a fund document (batched)
version: 1.0
variables:
  doc_name: ""
  pages_text: ""
---

**Document:** ${doc_name}

Summarize each page below in 1-2 sentences. Focus on what information or data is on each page.

Return ONLY valid JSON:

```json
{"pages": [{"page_num": 1, "summary": "..."}]}
```

${pages_text}
