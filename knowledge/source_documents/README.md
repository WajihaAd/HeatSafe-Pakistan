Put legally reusable text in `example.txt` and corresponding metadata in `example.json`, e.g.:

```json
{"title":"Verified document title","section":"Exact section","url":"https://official-source.example/document","organization":"Issuer","license":"Permission/licence details","keywords":"heat workers"}
```

Run `python streamlit_app/ingest_knowledge.py` and commit the generated `knowledge/ingested_chunks.json`.
Do not ingest copyrighted documents without permission; cite the real document and section.
