# 🔬 LabSentinel-RAG

**Multimodal Evidence-Grounded Retrieval for Automated Laboratory Incident Investigation**

> A multimodal Retrieval-Augmented Generation (RAG) system that autonomously investigates laboratory safety incidents from video footage, grounding every finding in institutional SOPs and comparable historical cases.

---

## Overview

Investigating laboratory safety incidents is currently a manual, time-consuming process. An investigator must review CCTV footage, cross-reference safety SOPs and equipment manuals, and search historical incident records.

**LabSentinel-RAG automates this entire workflow** using a 5-stage multimodal AI pipeline:

| Stage | Component | Description |
|-------|-----------|-------------|
| **1** | Input | Incident video + Lab SOPs + Equipment Manuals + Incident Records |
| **2** | Video Analysis | GPT-4o Vision extracts a timestamped event timeline |
| **3** | RAG Retrieval | ChromaDB retrieves relevant SOP clauses & similar past incidents |
| **4** | LLM Reasoning | GPT-4o confirms violations, explains root cause, recommends actions |
| **5** | Output | JSON + HTML + PDF investigation report |

---

## Key Features

- **Multimodal input**: Accepts CCTV video, images, textual incident descriptions
- **Evidence-grounded**: Every violation is linked to specific SOP clauses (not just model knowledge)
- **Timestamp-linked evidence**: Each finding includes the video frame where it was detected
- **Similar incident retrieval**: Finds comparable past cases for pattern recognition
- **Structured output**: Machine-readable JSON + human-readable HTML/PDF report
- **Autonomous**: Produces a complete investigation report from a single video input

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       LabSentinel-RAG                           │
│          Multimodal Evidence-Grounded Incident Investigation    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│   1. INPUT   │──▶│ 2. VIDEO     │──▶│  3. RETRIEVE │──▶│ 4. REASON    │──▶│  5. OUTPUT   │
│              │   │    ANALYSIS  │   │     (RAG)    │   │    (LLM)     │   │              │
│ • Video      │   │              │   │              │   │              │   │ • Timeline   │
│ • Images     │   │ • Frame      │   │ • SOPs       │   │ • Confirm    │   │ • Violations │
│ • SOPs       │   │   extraction │   │ • Manuals    │   │   violations │   │ • SOP refs   │
│ • Manuals    │   │ • VLM event  │   │ • Past       │   │ • Root cause │   │ • Past cases │
│ • Incidents  │   │   detection  │   │   incidents  │   │ • Actions    │   │ • Actions    │
└──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
                                              ▲
                                    ┌─────────────────┐
                                    │  KNOWLEDGE BASE  │
                                    │  (ChromaDB +     │
                                    │  SentenceXformrs)│
                                    └─────────────────┘
```

---

## Quick Start

### 1. Installation

```bash
cd labsentinel-rag
pip install -r requirements.txt
```

### 2. Configure API Key

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Build the Knowledge Base (one-time)

```bash
# Add your SOP documents to data/sops/
# Add equipment manuals to data/manuals/
# Add past incident records to data/incidents/

python main.py build-kb
```

### 4. Run an Investigation

```bash
python main.py investigate data/videos/incident.mp4
```

### 5. Try the Mock Demo (no API key needed)

```bash
python demo_mock.py
# Open outputs/report_*.html in your browser
```

### 6. Launch the Web UI

```bash
python app.py
# Open http://localhost:7860
```

---

## Project Structure

```
labsentinel-rag/
├── src/
│   ├── vlm/                    # Stage 2: Vision-Language Model video analysis
│   │   ├── frame_extractor.py  #   Temporal + motion-based frame sampling
│   │   └── video_analyzer.py   #   GPT-4o Vision per-frame + timeline analysis
│   ├── ingestion/              # Stage 0: Knowledge base construction
│   │   ├── document_loader.py  #   PDF/DOCX/TXT/JSON chunking
│   │   └── knowledge_base_builder.py  # ChromaDB indexing
│   ├── retrieval/              # Stage 3: RAG evidence retrieval
│   │   └── retriever.py        #   Hybrid SOP + manual + incident retrieval
│   ├── reasoning/              # Stage 4: LLM violation reasoning
│   │   └── violation_reasoner.py  # Evidence-grounded violation analysis
│   ├── report/                 # Stage 5: Report generation
│   │   └── report_generator.py #   JSON + HTML + PDF output
│   ├── utils/                  # Shared models and logging
│   │   ├── models.py           #   Pydantic data models
│   │   └── logger.py           #   Structured logging
│   └── pipeline.py             # Main orchestrator
├── data/
│   ├── sops/                   # Lab Safety SOPs
│   ├── manuals/                # Equipment manuals
│   ├── incidents/              # Historical incident records
│   ├── videos/                 # Input videos
│   └── embeddings/             # ChromaDB vector store
├── outputs/                    # Generated reports
├── tests/                      # Unit and integration tests
├── config/
│   └── settings.py             # Global configuration (Pydantic)
├── main.py                     # CLI entry point (Typer)
├── app.py                      # Gradio web demo
├── demo_mock.py                # Mock demo (no API key needed)
└── requirements.txt
```

---

## Configuration

Edit `config/settings.py` or set environment variables:

| Setting | Default | Description |
|---------|---------|-------------|
| `llm.model` | `gpt-4o` | VLM for video frame analysis |
| `video.frame_sample_rate` | `2` | Sample 1 frame per N seconds |
| `video.max_frames` | `30` | Maximum frames per video |
| `rag.embedding_model` | `all-MiniLM-L6-v2` | Sentence transformer model |
| `rag.chunk_size` | `512` | Words per document chunk |
| `rag.top_k_sops` | `3` | SOP clauses retrieved per event |
| `rag.top_k_incidents` | `3` | Historical incidents per event |

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Comparison with Existing Systems

| System | Domain | Timestamp Evidence | SOP Grounding | Historical Cases | Structured Report |
|--------|--------|--------------------|---------------|------------------|-------------------|
| AcciVid (2024) | Construction | ❌ | ❌ | ❌ | ❌ |
| TrafficRAG (2026) | Traffic/Legal | ❌ | ❌ | Partial | Static |
| AccidentGPT (2024) | Traffic | ❌ | ❌ | ❌ | ❌ |
| **LabSentinel-RAG** | **Laboratory** | **✅** | **✅** | **✅** | **✅** |

---

## Citation

```bibtex
@article{labsentinel2024,
  title={LabSentinel-RAG: Multimodal Evidence-Grounded Retrieval for 
         Automated Laboratory Incident Investigation},
  year={2024},
  abstract={A multimodal RAG system for automated laboratory incident 
            investigation that grounds every detected violation in 
            institutional SOP evidence and comparable historical cases.}
}
```

---

## Keywords

Multimodal RAG · Video Understanding · Vision-Language Models · Incident Investigation · Evidence Grounding · Temporal Reasoning · Laboratory Safety · Hybrid Retrieval
