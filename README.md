# Solution

Generated mock support dataset:
- `ai_school/`

Regenerate it with:
- `..\.venv\Scripts\python .\scripts\generate_ai_school_assets.py`

## Running locally

Set your OpenAI and LangSmith credentials (see `.env.example`), then:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Docker

Build and run the Streamlit UI inside Docker:

```bash
docker build -t sweet-attention .
docker run --env-file .env -p 8501:8501 sweet-attention
```
