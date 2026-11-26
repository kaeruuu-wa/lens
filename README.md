# Lens

**Understand documents faster with AI-powered text analysis and summarization.**

Lens helps students, researchers, and professionals get the point of a long document quickly. Paste text or upload a PDF, and Lens shows the key numbers, the words that matter, an AI-written summary, and the overall tone.

**Live demo:** https://lens-bscs3a-ai.streamlit.app/

---

## Features

- **Two ways to add a document:** paste text, or upload a PDF (Lens reports how many pages it could read)
- **Document overview:** word, sentence, character, and unique word counts, average words per sentence, and estimated reading time
- **Most frequent words:** a bar chart of the top 15 meaningful words, with common stop words removed
- **Keyword insights:** repeated two-word phrases first, then recurring single words
- **AI summary:** Short, Medium, or Detailed, with long documents split into chunks and combined
- **Sentiment analysis:** positive, negative, or neutral, with a confidence score (optional)
- **Friendly error handling:** scanned PDFs, damaged files, and text that is too short all get a clear message
- **Choice of summary model:** best quality (BART Large) or lighter and faster (DistilBART)

## Tools and AI models

| Purpose | Tool |
|---|---|
| Web app and UI | Python, Streamlit |
| PDF text extraction | pdfplumber |
| Charts | Plotly |
| Summarization | `facebook/bart-large-cnn` (lighter option: `sshleifer/distilbart-cnn-12-6`) |
| Sentiment | `distilbert-base-uncased-finetuned-sst-2-english` |
| AI runtime | Hugging Face Transformers + PyTorch (CPU) |
| Version control and hosting | Git, GitHub, Streamlit Community Cloud |

## How it works

1. **Reading the document.** Pasted text is cleaned up. PDFs are read page by page with pdfplumber. If no text can be found, Lens explains that the file is probably scanned.
2. **Statistics and keywords.** Plain Python counts words and sentences. Stop words are removed using a built-in list, so no extra downloads are needed. Keywords come from simple frequency analysis.
3. **Summary.** The text is split into chunks of about 450 words, breaking at sentence ends. Each chunk is summarized by the model, and the partial summaries are combined into a final one. For speed, at most 8 chunks are summarized, picked evenly across the document. Lens tells the user when this happens.
4. **Sentiment.** Several sections of the document are scored and averaged. The model only knows positive and negative, so Lens reports "Neutral" when the sections balance out.
5. **Speed.** Models are loaded once with `st.cache_resource`, summaries are cached per text, length, and model, and results are kept in `st.session_state` so they don't disappear when a setting changes.

## Project structure

```
lens/
├── app.py                  # Streamlit interface and main flow
├── requirements.txt        # Python dependencies
├── .gitignore
├── .streamlit/
│   └── config.toml         # Theme and server settings
└── utils/
    ├── __init__.py
    ├── pdf_reader.py       # PDF text extraction
    ├── text_analysis.py    # Stats, frequent words, keywords, chunking
    ├── summarizer.py       # Hugging Face summarization
    └── sentiment.py        # Hugging Face sentiment analysis
```

## Run it locally

You need Python 3.11 or 3.12.

```bash
git clone https://github.com/YOUR-USERNAME/lens.git
cd lens
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py
```

Then open http://localhost:8501.

The first analysis downloads the AI models (BART Large is about 1.6 GB), so do it on a good connection. After that, the models are cached on your machine.

If the terminal shows a long `torch.classes` error, it is a harmless Streamlit file-watcher warning. Hide it with:

```bash
python -m streamlit run app.py --server.fileWatcherType none
```

## Customizing

- **Colors:** change `PALETTE` near the top of `app.py` to `"Indigo"`, `"Teal"`, `"Ocean"`, `"Rose"`, or `"Amber"`. Update `primaryColor` in `.streamlit/config.toml` to match.
- **Default summary model:** change `DEFAULT_MODEL_LABEL` in `utils/summarizer.py`.
- **Summary lengths:** adjust `LENGTH_SETTINGS` in `utils/summarizer.py`.

## Deploying to Streamlit Community Cloud

1. Push the project to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **Create app**, choose the repository and the `main` branch, and set the main file to `app.py`.
4. Under **Advanced settings**, choose Python 3.11 or 3.12.
5. Click **Deploy**. The first build takes several minutes because it installs PyTorch.

Every `git push` to the repository redeploys the app automatically.

**If the app runs out of memory,** switch the default to the lighter model by setting `DEFAULT_MODEL_LABEL = "Fast: DistilBART (lighter)"` in `utils/summarizer.py`, then push.

**Optional Hugging Face token.** Lens works without one. If you hit download rate limits, add a read-only token in the app's **Settings → Secrets**:

```toml
HF_TOKEN = "hf_your_token_here"
```

Never put tokens in the code or commit them to GitHub. `.gitignore` already excludes `.streamlit/secrets.toml` and `.env`.

## Limitations

- Summaries are written by an AI model and can miss details or get things slightly wrong. Lens is a reading aid, not a replacement for reading.
- Very long documents are summarized from evenly spaced sections, not every word.
- Scanned or image-only PDFs are not supported, because Lens does not do OCR.
- The sentiment model was trained on movie reviews, so it works best on opinionated text and is less reliable on neutral, technical writing.
- The models work best on English text.
- On the public demo, text is processed on Streamlit's servers, so don't upload private documents.

## Ideas for next steps

- OCR support for scanned PDFs
- More languages
- Ask questions about a document
- Export results as a PDF report

## Credits

Built by
PERE, JOSHUA MICHAEL
SERGUEL, DEA FEI
IBAHAN, NATHANAEL

BSCS 3A

Models: BART by Meta AI, DistilBART and DistilBERT by Hugging Face, hosted through the Hugging Face Hub. Built with Streamlit, pdfplumber, Plotly, and PyTorch.
