# 🔬 LabSentinel-RAG

**Multimodal Evidence-Grounded Retrieval for Automated Laboratory Incident Investigation**

> Upload any lab manual + any CCTV/incident video → get a structured safety violation report with exact timestamps, cited rules, and corrective actions. Zero hardcoded rules — everything comes from your uploaded manual.

🔗 **Live Repo:** https://github.com/Suhani-Srikantaswamy/labsentinel-rag

---

## What It Does

LabSentinel-RAG investigates lab safety incidents automatically using a 5-stage AI pipeline:

| Stage | What Happens |
|-------|-------------|
| **1. Input** | Upload any lab manual (PDF/DOCX/TXT) + incident video (any length) |
| **2. Video Analysis** | Gemini Vision analyses every frame using a structured safety checklist |
| **3. RAG Retrieval** | Retrieves the most relevant passages from your manual for each event |
| **4. LLM Reasoning** | Confirms violations, cites the exact rule, gives corrective actions |
| **5. Report** | Timestamped HTML + PDF report — every violation linked to exact video second |

---

## Key Features

- **Zero hardcoded rules** — upload a chemistry manual → chemistry rules; biology manual → biology rules
- **Exact timestamps** — every finding shows the precise second in the video
- **Rule citation** — quotes the exact section from your manual
- **Any lab type** — chemistry, biology, cleanroom, hospital, industrial
- **Any video length** — processes hours of CCTV footage in chunks automatically
- **Multi-key rotation** — supports multiple Gemini API keys, rotates when one hits quota
- **Fully free** — runs on Google Gemini free tier (no credit card needed)

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/Suhani-Srikantaswamy/labsentinel-rag.git
cd labsentinel-rag
pip install -r requirements.txt
```

### 2. Get a FREE API Key

Go to **https://aistudio.google.com/apikey** → sign in with Google → Create API key

### 3. Configure

```bash
cp .env.example .env
```

Open `.env` and add your key:
```
GOOGLE_API_KEY=your-gemini-key-here
```

To add multiple keys for automatic rotation (recommended):
```
GOOGLE_API_KEY=key1
GOOGLE_API_KEY_2=key2
GOOGLE_API_KEY_3=key3
```

### 4. Build Knowledge Base (optional but improves accuracy)

```bash
python main.py build-kb
```

### 5. Launch the Web UI

```bash
python app.py
```

Open **http://localhost:7860**

---

## How to Use the Web UI

1. **Upload your lab manual** — PDF, DOCX, or TXT (any lab's safety manual)
2. **Upload a video** or paste a YouTube URL
3. Click **🚀 Run Investigation**
4. View the full report inline — download PDF with one click

---

## Sample Files Included

| File | Description |
|------|-------------|
| `sample_files/chemistry_lab_manual.txt` | Full chemistry lab safety manual (PPE, acids, fume hoods, waste) |
| `sample_files/biology_lab_manual.txt` | BSL-2 biology/microbiology lab manual (biosafety, BSC, sharps) |
| `scripts/create_sample_video.py` | Generates a synthetic 96s lab incident video with 6 violations |
| `data/sops/` | Sample SOPs for knowledge base |
| `data/incidents/` | Sample historical incident records |

---

## Verified Test Datasets

All links confirmed working:

| Video | Manual | What It Tests |
|-------|--------|--------------|
| [Fume Hood Safety (4min)](https://www.youtube.com/watch?v=PZRQMK6udZs) | [MIT Chemistry Manual](https://chemistry.mit.edu/wp-content/uploads/2026/01/2026-Chemical-Hygiene-Manual.pdf) | Fume hood, PPE |
| [Lab Safety DOs & DON'Ts (8min)](https://www.youtube.com/watch?v=saXFQR86ziM) | [MIT Safety Rules](https://web.mit.edu/5.310/www/safety.pdf) | Multiple violations |
| [Lab Safety Training (40min)](https://www.youtube.com/watch?v=TlhVR0xQLJQ) | [UC Davis Lab Manual](https://chemistry.ucdavis.edu/sites/g/files/dgvnsk196/files/inline-files/CHE%202B%20Student%20Lab%20Manual%20Spring%202023.pdf) | Full training |
| [Biosafety Cabinet (4min)](https://www.youtube.com/watch?v=KSH_aaQSNyE) | `sample_files/biology_lab_manual.txt` | Biological safety |

---

## Project Structure

```
labsentinel-rag/
├── app.py                  # Gradio web UI
├── main.py                 # CLI (build-kb, investigate, stats)
├── config/
│   └── settings.py         # All config via environment variables
├── src/
│   ├── vlm/                # Video frame extraction + Gemini Vision analysis
│   ├── ingestion/          # Document loading + ChromaDB indexing
│   ├── retrieval/          # RAG retrieval from knowledge base
│   ├── reasoning/          # LLM violation analysis + report generation
│   ├── report/             # HTML + PDF + JSON report rendering
│   └── utils/              # Models, logging, LLM client (key rotation)
├── data/
│   ├── sops/               # Your lab SOPs (add your own here)
│   ├── manuals/            # Equipment manuals
│   └── incidents/          # Historical incident records
├── sample_files/           # Ready-to-use sample lab manuals
├── scripts/                # Sample video generator
└── tests/                  # Unit tests (pytest)
```

---

## Configuration

All settings via `.env` — nothing hardcoded:

```bash
# Model
LABSENTINEL_GEMINI_MODEL=gemini-3.5-flash-lite   # vision model

# Video analysis
LABSENTINEL_FRAME_RATE=3          # analyse 1 frame every N seconds
LABSENTINEL_MAX_FRAMES=60         # total frame cap
LABSENTINEL_MOTION_THRESHOLD=10.0 # motion sensitivity

# Retrieval
LABSENTINEL_EMBED_MODEL=all-MiniLM-L6-v2
LABSENTINEL_TOP_K_SOPS=5
LABSENTINEL_SIM_THRESHOLD=0.20

# Output
LABSENTINEL_OUTPUT_FORMATS=json,html,pdf
```

---

## Free Tier Limits

| Provider | Cost | Limit |
|----------|------|-------|
| Google Gemini | **Free** | ~20 req/day per key |
| With 4 keys | **Free** | ~80 req/day = ~5 full investigations/day |
| OpenAI GPT-4o | ~$0.30/run | Unlimited |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Acknowledgements

Built with: Google Gemini, ChromaDB, SentenceTransformers, Gradio, LangChain, ReportLab
