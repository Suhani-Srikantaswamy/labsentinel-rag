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
| **3. RAG Retrieval** | Indexes your uploaded manual and retrieves its most relevant passages (plus the knowledge base) for each event |
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
- **Evidence screenshots** — every violation shows the exact video frame (HTML + PDF)
- **Click-to-play timestamps** — click any timestamp, chart marker or screenshot to jump the evidence player to that moment
- **Violation timeline chart** — severity-coloured markers across the whole video
- **Human review** — ✓ Confirm / ✗ False alarm on each violation, saved into the report
- **No repeat flags** — each person is tracked across frames (P1, P2 …); once someone is flagged for an issue, later sightings are listed under "Also seen at" instead of becoming new violations

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

Open **http://localhost:7860** (or run `python main.py demo`)

### 6. Or use the CLI

```bash
python main.py investigate path/to/video.mp4 --manual path/to/lab_manual.pdf
```

### Or run with Docker

```bash
cp .env.example .env        # add your GOOGLE_API_KEY
docker compose up --build   # → http://localhost:7860
```

`data/` and `outputs/` are mounted as volumes, and API keys come from `.env` at runtime (they are never baked into the image).

---

## How to Use the Web UI

1. **Upload your lab manual** — PDF, DOCX, or TXT (any lab's safety manual)
2. **Upload a video** or paste a YouTube URL
3. Click **🚀 Run Investigation**
4. Review the report inline:
   - Click a **timestamp**, **timeline marker** or **screenshot** → the Evidence Player jumps to that moment
   - Mark each violation **✓ Confirm** or **✗ False alarm** (↺ Undo to reset) — saved to the report JSON, HTML and PDF
5. Download the PDF with one click. The standalone HTML report in `outputs/` has its own video player when opened locally.

---

## Sample Files Included

| File | Description |
|------|-------------|
| `data/sops/` | Sample SOPs (general lab safety, chemical handling) — also usable as an uploaded manual |
| `data/manuals/` | Sample equipment manual (centrifuge) |
| `data/incidents/` | Sample historical incident records |

---

## Verified Test Datasets

All links confirmed working:

| Video | Manual | What It Tests |
|-------|--------|--------------|
| [Fume Hood Safety (4min)](https://www.youtube.com/watch?v=PZRQMK6udZs) | [MIT Chemistry Manual](https://chemistry.mit.edu/wp-content/uploads/2026/01/2026-Chemical-Hygiene-Manual.pdf) | Fume hood, PPE |
| [Lab Safety DOs & DON'Ts (8min)](https://www.youtube.com/watch?v=saXFQR86ziM) | [MIT Safety Rules](https://web.mit.edu/5.310/www/safety.pdf) | Multiple violations |
| [Lab Safety Training (40min)](https://www.youtube.com/watch?v=TlhVR0xQLJQ) | [UC Davis Lab Manual](https://chemistry.ucdavis.edu/sites/g/files/dgvnsk196/files/inline-files/CHE%202B%20Student%20Lab%20Manual%20Spring%202023.pdf) | Full training |
| [Biosafety Cabinet (4min)](https://www.youtube.com/watch?v=KSH_aaQSNyE) | Any BSL-2 biosafety manual | Biological safety |

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
│   ├── report/             # JSON + PDF rendering; HTML via templates/report.html.j2
│   └── utils/              # Models, logging, LLM client (key rotation)
├── data/
│   ├── sops/               # Your lab SOPs (add your own here)
│   ├── manuals/            # Equipment manuals
│   └── incidents/          # Historical incident records
└── tests/                  # Unit tests (pytest)
```

---

## Configuration

All settings via `.env` — nothing hardcoded:

```bash
# Model
LABSENTINEL_GEMINI_MODEL=gemini-3.5-flash        # vision model

# Reliability
LABSENTINEL_THINKING_HEADROOM=3072  # output tokens reserved for model "thinking" (prevents cut-off JSON)
LABSENTINEL_REQUEST_TIMEOUT=120     # seconds per API call
LABSENTINEL_TRANSIENT_RETRIES=4     # retries on 503 / timeouts

# Video analysis
LABSENTINEL_FRAME_RATE=6          # minimum spacing of the evenly spaced frames (s)
LABSENTINEL_MAX_FRAMES=auto       # auto = 1 frame / 10 s, between 24 and 90 frames

# Speed
LABSENTINEL_FRAMES_PER_REQUEST=6  # consecutive frames analysed in one request
LABSENTINEL_EVENTS_PER_REQUEST=5  # flagged moments verified in one request
LABSENTINEL_PARALLEL_REQUESTS=8   # requests in flight (≈3 per usable key)
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

`tests/test_e2e.py` runs the whole pipeline (frame sampling → retrieval → reasoning → reports → review) on a generated video with a fake LLM, so it uses no API quota.

---

## Acknowledgements

Built with: Google Gemini, ChromaDB, SentenceTransformers, Gradio, ReportLab
