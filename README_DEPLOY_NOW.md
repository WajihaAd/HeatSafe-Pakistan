# HeatSafe Pakistan — Dashboard + RAG repair

This patch replaces only `streamlit_app/app.py` and `streamlit_app/rag_engine.py` on your **rag-dashboard-development** branch. Keep all existing `src/`, `models/`, `flows/`, `data/`, `knowledge/`, and other files.

## Browser-only installation

1. Open your fork `WajihaAd/HeatSafe-Pakistan` and switch to `rag-dashboard-development`.
2. Open `streamlit_app/app.py` → pencil (Edit) → replace the entire file with the patch's `streamlit_app/app.py` → Commit changes to this branch.
3. Repeat for `streamlit_app/rag_engine.py`.
4. Do not replace `streamlit_app/requirements.txt` or your existing `knowledge/guidance.json`.
5. Open your existing Streamlit app, use Manage app → Reboot if it has not automatically redeployed.
6. Test Ask HeatSafe AI with **What is the temperature in Karachi now?**, **What precautions should outdoor workers take?**, **What is the highest historical temperature?**, and **What is the current heat risk prediction?**

## What was fixed

- Previously, live-weather intent matching only recognized a few exact phrases. `the temp of Karachi is` fell through to guidance retrieval and abstained. The new routing handles more natural current-weather questions.
- Weather questions are answered from Open-Meteo, historical questions from the real processed CSV with supporting row IDs, and guidance questions from the existing knowledge JSON with document/section/source links.
- Overview has improved dark violet/blue styling, forecast chart, live metric cards, plain-language insights and a built-in Ask HeatSafe form.
- Weather Trends provides historical summary metrics, interactive time-series and extrema.
- Data Quality distinguishes historical checks, live freshness and missing Prefect publishing.

## Honest limitations

- The guidance knowledge base is still a **small manually summarized corpus**, not a fully verified set of indexed original official documents. Add licensed primary documents and verify sections for hackathon compliance.
- Retrieval uses TF-IDF vectors by default. This is a retrieval-grounded assistant; without a configured `GROQ_API_KEY`, answers are evidence summaries rather than LLM-generated conversation. A free API quota is not guaranteed.
- Hourly Prefect outputs still are not persisted into the Streamlit app. Live observations are fetched independently. The dashboard clearly labels this.
- The code compiles and retrieval was exercised with sample queries against the provided knowledge base; live hosted deployment and model deserialization have not been verified here.
- Do not merge into `main` until testing.
