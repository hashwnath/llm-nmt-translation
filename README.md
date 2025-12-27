# LLM-Augmented Machine Translation for E-Commerce Search

> **Paper Implementation**: This project implements the Multi-Locale Query Translation System described in the research paper on LLM-Augmented Machine Translation for Cross-Lingual E-Commerce Search.

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📄 Paper Attribution

This implementation is based on the research paper:
- **Title**: LLM-Augmented Machine Translation for Cross-Lingual E-Commerce Search (https://sigir-ecom.github.io/eCom25Papers/paper_18.pdf)
- **Architecture**: Multi-Locale Query Translation System (Figure 1)

## 🎯 Features

- **Language Detection**: "Is English?" check to skip translation for English queries
- **Translation Memory**: Fast cache lookup with <1ms latency for repeated queries
- **Entity-Aware Translation**: Preserves 50+ brand names (Liberté ≠ Liberty, Pampers, etc.)
- **Ambiguity Resolution**: Context-aware disambiguation using session history + GPT-4o-mini
- **Neural Machine Translation**: MarianMT with contextual post-processing rules
- **Offline LLM Preprocessing**: Pre-populates Translation Memory before runtime

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                   USER QUERY (French)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   LANGUAGE DETECTION        │
              │   "Is English?" → Skip      │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │  TIER 1: TRANSLATION        │
              │  MEMORY (Cache Layer)       │
              │  ✓ <1ms latency             │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │ TIER 2: ENTITY + AMBIGUITY  │
              │ ✓ Brand preservation        │
              │ ✓ Context-aware resolution  │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │ TIER 3: NMT + RULES         │
              │ ✓ MarianMT translation      │
              │ ✓ Post-processing fixes     │
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   OUTPUT: English Query     │
              └─────────────────────────────┘
```

## 🚀 Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/hashwnath/llm-nmt-translation.git
cd llm-nmt-translation

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 3. Run Offline Preprocessing (Optional)

```bash
python offline_preprocess.py
```

### 4. Start the Server

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Open the Demo

Visit [http://localhost:8000](http://localhost:8000)

## 🧪 Demo Examples

### Ambiguity Resolution

| Context | French Query | Translation |
|---------|--------------|-------------|
| 🍑 Fruits | `pêche fraîche` | fresh **peach** |
| 🎣 Sports | `pêche fraîche` | fresh **fishing** |
| 💄 Cosmetics | `parfum trésor` | **Tresor** perfume |
| 🥛 Dairy | `yogurt liberté` | **liberté** yogurt |

### Entity Preservation

| French Query | Translation | Preserved |
|--------------|-------------|-----------|
| `acheter Pampers` | buy Pampers | ✅ Brand |
| `papier Royale` | Royale paper | ✅ Brand |
| `dentifrice Colgate` | Colgate toothpaste | ✅ Brand |

## 📊 Performance

| Metric | Value |
|--------|-------|
| Cache Latency | <1ms |
| NMT Latency | ~500ms |
| Cache Hit Rate | 80-90% (with preprocessing) |

## 📁 Project Structure

```
llm-nmt-translation/
├── app/
│   ├── main.py               # FastAPI application
│   ├── config.py             # Configuration
│   ├── models.py             # Pydantic models
│   └── translation/
│       ├── pipeline.py       # Main orchestrator
│       ├── tier1_cache.py    # Translation Memory
│       ├── tier2_entity.py   # Entity Extractor
│       ├── tier2_ambiguity.py # Ambiguity Resolver
│       ├── tier3_nmt.py      # NMT Translator
│       └── metrics.py        # Metrics tracking
├── static/                   # Web UI
├── offline_preprocess.py     # LLM preprocessing
├── requirements.txt
└── README.md
```

## 🛠️ Technology Stack

- **Backend**: FastAPI, Uvicorn
- **NMT Model**: MarianMT (Helsinki-NLP/opus-mt-fr-en)
- **LLM**: OpenAI GPT-4o-mini (for ambiguity resolution)
- **Frontend**: Vanilla HTML/CSS/JS with modern dark theme

## 📝 License

MIT License

## 🙏 Acknowledgments

- Paper authors for the Multi-Locale Query Translation System architecture
- Helsinki-NLP for the MarianMT translation model
- OpenAI for GPT-4o-mini API
