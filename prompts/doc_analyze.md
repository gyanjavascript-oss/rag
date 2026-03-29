---
name: Document Analyzer
description: Classifies a fund document type, generates a summary, and extracts key topics
version: 1.0
variables:
  filename: ""
  sample: ""
  doc_types: ""
---

Analyze this fund document excerpt and return a JSON object.

**Filename:** ${filename}

**Document excerpt:**
${sample}

## Response Format

Return ONLY valid JSON with these exact keys:

```json
{
  "doc_type": "one of: ${doc_types}",
  "summary": "3-5 sentence summary of what this document covers, its purpose, and key terms or figures",
  "key_topics": ["topic1", "topic2", "topic3", "topic4", "topic5"]
}
```
