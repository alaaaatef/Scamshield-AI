# ScamShield AI — 90-Minute Build Plan

A RAG chatbot that answers phishing/scam questions using only your trusted PDFs,
built entirely with **free** tools:

| Piece | Tool | Cost |
|---|---|---|
| Embeddings | `sentence-transformers` (runs on your machine) | Free, no key |
| Vector DB | FAISS | Free, local |
| LLM | Groq API (Llama 3.3 70B) | Free tier, no card required |
| Hosting | Streamlit Community Cloud | Free |
| Code hosting | GitHub | Free |

Total time: ~90 minutes. Timings below are targets, not hard limits.

---

## Step 0 (5 min) — Install Python tools locally

You need Python 3.10+ installed. Then, in a terminal:

```bash
mkdir scamshield && cd scamshield
# copy in the files from this project: app.py, ingest.py, requirements.txt, .gitignore
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Step 1 (10 min) — Get a free Groq API key

1. Go to **https://console.groq.com/keys**
2. Sign up (free, no credit card).
3. Click "Create API Key" and copy it.
4. Create a local secrets file so the app can find it while testing:

```bash
mkdir -p .streamlit
echo 'GROQ_API_KEY = "paste-your-key-here"' > .streamlit/secrets.toml
```

## Step 2 (10 min) — Add your PDFs

```bash
mkdir -p data
# copy your PDFs into data/, e.g.:
# data/Phishing_Guidance_-_Stopping_the_Attack_Cycle_at_Phase_One_508c.pdf
# data/Avoiding-Social-Engineering-and-Phishing-Attacks.pdf
# data/emailscams_0905.pdf
# data/p144401_protecting_older_consumers_2019_1.pdf
# data/Phishing_actors_exploit_complex_routing...pdf
```

(You already have these 6 PDFs from your original project brief — just drop them into `data/`.)

## Step 3 (10 min) — Build the vector index

```bash
python ingest.py
```

This reads every PDF in `data/`, splits it into chunks, embeds them locally
(first run downloads a small ~80MB model, cached after that), and saves
`vector_store/index.faiss` + `vector_store/chunks.pkl`.

You should see output like:
```
Found 6 PDF(s): [...]
Created 480 chunks total. Embedding them now...
Done! Saved index + 480 chunks to 'vector_store/'.
```

## Step 4 (15 min) — Run and test locally

```bash
streamlit run app.py
```

Open the local URL it prints (usually `http://localhost:8501`). Try questions like:

- "I got an email saying my password expires today with a 'Stay Connected' button — is this phishing?"
- "Someone called claiming to be Social Security and said I owe money — what should I do?"
- "What is business email compromise?"

Check that answers include the structured sections (Attack Type, Risk Level, etc.)
and cite real source PDFs in the expander.

## Step 5 (15 min) — Push to GitHub

```bash
git init
git add .
git commit -m "ScamShield AI - RAG phishing assistant"
```

Create a new **public** repo on github.com (e.g. `scamshield-ai`), then:

```bash
git remote add origin https://github.com/YOUR-USERNAME/scamshield-ai.git
git branch -M main
git push -u origin main
```

**Important:** `vector_store/` must be pushed too (it's small — a few MB), so
the deployed app doesn't need to re-run `ingest.py` in the cloud. Do NOT push
`.streamlit/secrets.toml` (it's already in `.gitignore`) — you'll set the key
directly in Streamlit Cloud instead.

## Step 6 (15 min) — Deploy for free on Streamlit Community Cloud

1. Go to **https://share.streamlit.io** and sign in with GitHub (free).
2. Click "New app" → pick your `scamshield-ai` repo → branch `main` → main file `app.py`.
3. Before clicking Deploy, open **Advanced settings → Secrets** and paste:
   ```
   GROQ_API_KEY = "paste-your-key-here"
   ```
4. Click **Deploy**. Wait 2-3 minutes for the build.
5. You'll get a public URL like `https://scamshield-ai.streamlit.app` — free to
   share with anyone.

## Step 7 (10 min buffer) — Polish

- Add a short description + the public URL to your GitHub repo's "About" section.
- Optionally rename the Streamlit app URL in app settings.
- Test the live link on your phone to confirm it works for others.

---

## Notes / troubleshooting

- **Free Groq limits**: generous requests/day on the free tier, plenty for a demo
  or small class project. If you hit a rate limit, wait a minute or switch the
  `GROQ_MODEL` in `app.py` to `"llama-3.1-8b-instant"` (faster, smaller, higher
  free-tier throughput).
- **Alternative free LLM APIs** if you ever want to swap Groq out: Google
  Gemini API (free tier), or Hugging Face Inference API (free tier, smaller
  models). The `app.py` code only needs the `client.chat.completions.create(...)`
  call changed to whichever SDK you use.
- **No GPU needed** — `all-MiniLM-L6-v2` runs fine on CPU for this dataset size.
- If Streamlit Cloud build fails on `faiss-cpu`, it's almost always a Python
  version mismatch — add a `runtime.txt` file with `python-3.11` in the repo root.
