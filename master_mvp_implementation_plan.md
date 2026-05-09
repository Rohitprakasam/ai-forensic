# RAW — AI-Powered Unified Investigation OS
# Master MVP Implementation Plan (Featherless AI Edition)

---

## 🔑 Key Change: Featherless AI Instead of Local Ollama

**Original plan:** Self-hosted Ollama + Llama 3 (requires GPU, Docker, slow cold starts)
**This plan:** Featherless AI serverless API (OpenAI-compatible, zero infra, instant access to 30K+ models)

### Featherless AI Configuration

```python
# config/settings.py
import os

FEATHERLESS_API_KEY = os.getenv(
    "FEATHERLESS_API_KEY",
    "rc_b6cb34a0e41abcd2266da87e740d1ec579d172e3412d675051b0bd1b73bac7c1"
)
FEATHERLESS_BASE_URL = "https://api.featherless.ai/v1"

# Model Selection Strategy
MODELS = {
    "primary": "Qwen/Qwen3-32B",           # Main reasoning model (tool calling support)
    "fast": "Qwen/Qwen2.5-7B-Instruct",    # Fast responses (copilot chat)
    "embedding": "Qwen/Qwen3-Embedding-8B", # Embeddings via Featherless API
    "agent": "NousResearch/Hermes-3-Llama-3.1-8B",  # Agent tasks
}
```

### Unified LLM Client (Replaces ALL Ollama Calls)

```python
# services/llm_client.py
from openai import OpenAI

client = OpenAI(
    base_url="https://api.featherless.ai/v1",
    api_key=FEATHERLESS_API_KEY
)

def chat_completion(messages: list, model: str = None, temperature: float = 0.3, max_tokens: int = 2048) -> str:
    """Universal LLM call — replaces every Ollama request in the original plan."""
    response = client.chat.completions.create(
        model=model or MODELS["primary"],
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens
    )
    return response.choices[0].message.content

def generate_embeddings(text: str) -> list:
    """Generate embeddings via Featherless API — replaces local SentenceTransformer."""
    response = client.embeddings.create(
        model=MODELS["embedding"],
        input=text
    )
    return response.data[0].embedding

def chat_completion_with_tools(messages: list, tools: list, model: str = None) -> dict:
    """Tool-calling support for OpenClaw agents — native on Qwen3 family."""
    response = client.chat.completions.create(
        model=model or MODELS["primary"],
        messages=messages,
        tools=tools,
        max_tokens=4096
    )
    return response.choices[0].message
```

---

## 📌 Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    FRONTEND (Member 1)               │
│            React Dashboard + Graph Viz               │
└──────────────────────┬──────────────────────────────┘
                       │ REST API
┌──────────────────────▼──────────────────────────────┐
│              FASTAPI GATEWAY (:8001)                 │
│   /ai/*  /cctv/*  /graph/*  /search/*  /copilot/*   │
└───┬────────┬────────┬────────┬────────┬─────────────┘
    │        │        │        │        │
    ▼        ▼        ▼        ▼        ▼
┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐
│ OCR/ │ │ CCTV │ │Graph │ │Search│ │Copilot│
│ NLP  │ │Analytics│ │Intel │ │Engine│ │ Chat │
└──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘ └──┬───┘
   │        │        │        │        │
   └────────┴────────┴────────┴────────┘
                     │
        ┌────────────▼────────────┐
        │   FEATHERLESS AI API    │
        │  api.featherless.ai/v1  │
        │  (Qwen3-32B, Hermes,   │
        │   Qwen3-Embedding-8B)  │
        └─────────────────────────┘
                     +
    ┌──────┐  ┌──────┐  ┌──────────┐
    │Neo4j │  │Milvus│  │PostgreSQL│
    │:7474 │  │:19530│  │(Member 1)│
    └──────┘  └──────┘  └──────────┘
```

---

## 🗂️ Folder Structure

```
backend/
├── app/
│   ├── config/
│   │   └── settings.py              # Featherless API keys, model configs
│   │
│   ├── services/
│   │   ├── llm_client.py            # [NEW] Unified Featherless AI client
│   │   ├── ocr_service.py           # PaddleOCR + PyMuPDF text extraction
│   │   ├── nlp_service.py           # Summarization + NER (uses llm_client)
│   │   ├── cctv_service.py          # YOLOv8 + DeepSORT video analysis
│   │   ├── graph_service.py         # Neo4j graph intelligence
│   │   ├── search_service.py        # Milvus semantic search (uses llm_client for embeddings)
│   │   ├── risk_service.py          # Rule-based risk scoring
│   │   └── voice_service.py         # faster-whisper transcription
│   │
│   ├── ai/
│   │   ├── openclaw/
│   │   │   ├── agent_runner.py      # Core agent runner (uses llm_client)
│   │   │   ├── tool_registry.py     # Tool definitions for Featherless tool calling
│   │   │   └── prompts/
│   │   │       ├── evidence_prompt.txt
│   │   │       ├── timeline_prompt.txt
│   │   │       ├── legal_prompt.txt
│   │   │       └── copilot_prompt.txt
│   │   │
│   │   ├── evidence_agent.py
│   │   ├── timeline_agent.py
│   │   ├── cctv_agent.py
│   │   ├── graph_agent.py
│   │   └── legal_report_agent.py
│   │
│   ├── routes/
│   │   ├── ai_routes.py             # /ai/analyze/evidence
│   │   ├── cctv.py                  # /cctv/analyze, /cctv/events
│   │   ├── graph.py                 # /graph/case, /graph/suspect
│   │   ├── search.py                # /search/semantic, /search/keyword
│   │   ├── chatbot.py               # /copilot/ask
│   │   ├── voice.py                 # /voice/transcribe
│   │   └── agents.py                # /agents/evidence, /agents/timeline
│   │
│   ├── workers/
│   │   └── ai_tasks.py              # Celery async tasks
│   │
│   └── main.py                      # FastAPI app entry point
│
├── docker-compose.yml
├── requirements.txt
├── .env
└── README.md
```

---

## 📦 Requirements & Setup

### requirements.txt

```
# Core API
fastapi==0.115.0
uvicorn==0.30.0
python-dotenv==1.0.1
httpx==0.27.0
pydantic==2.9.0

# Featherless AI (OpenAI-compatible client)
openai==1.50.0

# OCR & Document Processing
paddlepaddle==2.6.0
paddleocr==2.8.0
PyMuPDF==1.24.0

# NLP
spacy==3.7.0

# Computer Vision
opencv-python==4.10.0
ultralytics==8.2.0
deep-sort-realtime==1.3.0

# Databases
neo4j==5.23.0
pymilvus==2.4.0

# Task Queue
celery==5.4.0
redis==5.0.0

# Voice
faster-whisper==1.0.0
```

### .env

```bash
FEATHERLESS_API_KEY=rc_b6cb34a0e41abcd2266da87e740d1ec579d172e3412d675051b0bd1b73bac7c1
FEATHERLESS_BASE_URL=https://api.featherless.ai/v1

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=investigation123

MILVUS_HOST=localhost
MILVUS_PORT=19530

REDIS_URL=redis://localhost:6379/0
MEMBER1_API=http://localhost:8000
```

### Docker Services (only databases — NO Ollama needed!)

```yaml
version: "3.8"
services:
  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/investigation123
    volumes:
      - neo4j_data:/data

  milvus-etcd:
    image: quay.io/coreos/etcd:v3.5.5
    environment:
      ETCD_AUTO_COMPACTION_MODE: revision
      ETCD_AUTO_COMPACTION_RETENTION: "1000"
      ETCD_QUOTA_BACKEND_BYTES: "4294967296"
    command: etcd -advertise-client-urls=http://127.0.0.1:2379 -listen-client-urls http://0.0.0.0:2379

  milvus-minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /minio_data

  milvus:
    image: milvusdb/milvus:v2.4.0
    ports:
      - "19530:19530"
    depends_on:
      - milvus-etcd
      - milvus-minio

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  neo4j_data:
```

> **💡 Major Win:** Original plan required Ollama Docker container + GPU + pulling 4GB+ models.
> Now we just call `api.featherless.ai` — zero GPU, zero Docker for LLM, instant start.

---

## 🔧 Module-by-Module Implementation

### Module 1 — OCR & NLP Pipeline

**No changes** to `ocr_service.py` (PaddleOCR stays local — it's fast and free).

**`services/nlp_service.py`** — Rewritten to use Featherless:

```python
from app.services.llm_client import chat_completion, generate_embeddings
import spacy

nlp = spacy.load("en_core_web_sm")

def summarize_text(text: str, context: str = "forensic evidence") -> str:
    messages = [
        {"role": "system", "content": f"""You are a forensic AI analyst.
Analyze the following {context} document.
Extract: key facts, persons mentioned, locations, dates, and suspicious elements.
Provide a structured summary in 5-8 bullet points."""},
        {"role": "user", "content": text[:4000]}
    ]
    return chat_completion(messages)

def extract_entities(text: str) -> dict:
    doc = nlp(text)
    return {
        "persons": list({ent.text for ent in doc.ents if ent.label_ == "PERSON"}),
        "locations": list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}),
        "dates": list({ent.text for ent in doc.ents if ent.label_ == "DATE"}),
        "organizations": list({ent.text for ent in doc.ents if ent.label_ == "ORG"}),
        "money": list({ent.text for ent in doc.ents if ent.label_ == "MONEY"}),
    }

# Embeddings now use Featherless API instead of local SentenceTransformer
# generate_embeddings() is imported from llm_client
```

### Module 2 — CCTV Analytics

**No changes** — YOLOv8 + DeepSORT run locally (computer vision, not LLM). Same code as original plan.

### Module 3 — Graph Intelligence (Neo4j)

**No changes** — Neo4j queries are Cypher, not LLM. Same code as original plan.

### Module 4 — Semantic Search (Milvus)

**Key change:** Embeddings now come from Featherless API (`Qwen/Qwen3-Embedding-8B`).

```python
# services/search_service.py
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from app.services.llm_client import generate_embeddings

connections.connect("default", host="localhost", port="19530")
COLLECTION_NAME = "evidence_embeddings"

def init_milvus():
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="evidence_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=4096),
            # NOTE: dim=4096 for Qwen3-Embedding-8B (not 384 like MiniLM)
        ]
        schema = CollectionSchema(fields)
        collection = Collection(COLLECTION_NAME, schema)
        collection.create_index("embedding", {
            "index_type": "IVF_FLAT",
            "metric_type": "COSINE",
            "params": {"nlist": 128}
        })
    return Collection(COLLECTION_NAME)

def store_embedding(evidence_id: str, embedding: list, metadata: dict):
    collection = init_milvus()
    collection.insert([
        [evidence_id],
        [metadata.get("case_id", "")],
        [metadata.get("summary", "")],
        [embedding]
    ])
    collection.flush()

def semantic_search(query: str, case_id: str = None, top_k: int = 5) -> list:
    query_embedding = generate_embeddings(query)  # Featherless API call
    collection = init_milvus()
    collection.load()

    search_params = {"metric_type": "COSINE", "params": {"nprobe": 10}}
    expr = f'case_id == "{case_id}"' if case_id else None

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=expr,
        output_fields=["evidence_id", "summary"]
    )

    return [
        {
            "evidence_id": r.entity.get("evidence_id"),
            "summary": r.entity.get("summary"),
            "relevance_score": round(r.score, 4)
        }
        for r in results[0]
    ]
```

### Module 5 — Risk Scoring

**No changes** — Pure Python rule-based logic. Same code as original plan.

### Module 6 — Voice Service

**No changes** — `faster-whisper` runs locally. Same code as original plan.

### Module 7 — OpenClaw AI Agents (Rewritten for Featherless)

```python
# ai/openclaw/agent_runner.py
from app.services.llm_client import chat_completion, chat_completion_with_tools
from app.config.settings import MODELS

def run_agent(agent_name: str, system_prompt: str, context: str, model: str = None) -> str:
    """Core agent runner — uses Featherless AI chat completions."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"CASE CONTEXT:\n{context}\n\nProvide your structured forensic analysis. Be specific, factual, and actionable."}
    ]
    return chat_completion(messages, model=model or MODELS["primary"])

def run_agent_with_tools(agent_name: str, system_prompt: str, context: str, tools: list) -> dict:
    """Agent runner with tool calling — uses Qwen3 native tool support."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": context}
    ]
    return chat_completion_with_tools(messages, tools, model=MODELS["primary"])
```

```python
# ai/openclaw/tool_registry.py
"""Tool definitions for Featherless native tool calling (Qwen3 family)."""

INVESTIGATION_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_evidence",
            "description": "Semantic search across all case evidence",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural language search query"},
                    "case_id": {"type": "string", "description": "Case ID to search within"}
                },
                "required": ["query", "case_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_suspect_connections",
            "description": "Get relationship graph connections for a suspect",
            "parameters": {
                "type": "object",
                "properties": {
                    "person_name": {"type": "string", "description": "Name of the suspect"}
                },
                "required": ["person_name"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_timeline_events",
            "description": "Get chronological events for a case",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"},
                    "start_time": {"type": "string", "description": "ISO timestamp"},
                    "end_time": {"type": "string", "description": "ISO timestamp"}
                },
                "required": ["case_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_risk_score",
            "description": "Calculate risk assessment score for a case",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_id": {"type": "string"}
                },
                "required": ["case_id"]
            }
        }
    }
]
```

### Module 8 — Investigation Copilot (Rewritten for Featherless)

```python
# routes/chatbot.py
from fastapi import APIRouter
from pydantic import BaseModel
from app.services.llm_client import chat_completion
from app.config.settings import MODELS
import httpx

router = APIRouter(prefix="/copilot", tags=["Investigation Copilot"])

class CopilotQuery(BaseModel):
    case_id: str
    question: str
    conversation_history: list = []

@router.post("/ask")
async def ask_copilot(query: CopilotQuery):
    case_context = await build_case_context(query.case_id)

    messages = [
        {"role": "system", "content": f"""You are RAW — an AI-powered forensic investigation copilot.
You have full access to the case intelligence below. Answer the investigator's question
concisely and accurately. If you do not have enough data, say so honestly. Never fabricate facts.

CASE INTELLIGENCE:
{case_context}"""}
    ]

    # Add conversation history
    for msg in query.conversation_history[-6:]:
        messages.append({"role": "user", "content": msg["question"]})
        messages.append({"role": "assistant", "content": msg["answer"]})

    messages.append({"role": "user", "content": query.question})

    # Use fast model for responsive chat
    answer = chat_completion(messages, model=MODELS["fast"])
    return {"answer": answer, "case_id": query.case_id}

async def build_case_context(case_id: str) -> str:
    base = "http://localhost:8000"
    async with httpx.AsyncClient() as client:
        case = (await client.get(f"{base}/cases/{case_id}")).json()
        evidence = (await client.get(f"{base}/evidence/case/{case_id}")).json()
        timeline = (await client.get(f"{base}/timeline/{case_id}")).json()

    context = f"Case: {case.get('title')} | Crime: {case.get('crime_type')} | Status: {case.get('status')}\n"
    context += f"Evidence items: {len(evidence)}\nTimeline events: {len(timeline)}\n"
    for ev in evidence[:5]:
        context += f"- Evidence: {ev.get('type')} | Summary: {ev.get('ai_summary', 'N/A')}\n"
    return context
```

---

## 🔌 API Contracts

```
POST   /ai/analyze/evidence             → Trigger AI analysis on evidence
POST   /cctv/analyze/{case_id}          → Upload + analyze CCTV footage
GET    /cctv/events/{case_id}           → Get CCTV events
GET    /graph/case/{case_id}            → Full relationship graph
GET    /graph/suspect/{name}            → Suspect connections
GET    /graph/mastermind/{case_id}      → Most connected suspects
POST   /search/semantic                 → Semantic search (Featherless embeddings)
GET    /search/keyword                  → Keyword search
POST   /voice/transcribe/{case_id}     → Transcribe audio
POST   /agents/evidence/{case_id}      → Run evidence agent
POST   /agents/timeline/{case_id}      → Run timeline agent
POST   /agents/legal-report/{case_id}  → Generate legal report
GET    /risk/score/{case_id}           → Risk score
POST   /copilot/ask                    → Investigation copilot chat
```

---

## 🎯 MVP Demo Flow

```
1. Upload PDF evidence
   → PaddleOCR extracts text → Featherless Qwen3-32B summarizes
   → Featherless Qwen3-Embedding-8B generates embeddings → stored in Milvus
   → spaCy NER → entities auto-added to Neo4j graph

2. Upload CCTV footage
   → YOLOv8 detects persons → DeepSORT tracks → events logged

3. Ask: "Who are the connected suspects?"
   → Neo4j graph query → returns suspect network

4. Copilot chat: "What happened between 9pm and 11pm?"
   → Featherless Qwen2.5-7B answers using timeline + evidence context

5. Semantic search: "Show me financial evidence"
   → Featherless embedding → Milvus cosine search → top results

6. Risk Score calculated → Case flagged as HIGH

7. Legal Report Agent → Featherless Qwen3-32B generates full report

8. DEMO COMPLETE ✅
```

---

## ⚡ Benefits of Featherless AI Over Ollama

| Aspect | Ollama (Original) | Featherless AI (This Plan) |
|---|---|---|
| Setup time | Pull 4GB+ models, need GPU | Zero setup, API key only |
| GPU required | Yes (or very slow on CPU) | No — serverless inference |
| Model quality | Llama 3 8B (limited) | Qwen3-32B, 30K+ models |
| Embeddings | Local SentenceTransformer | Qwen3-Embedding-8B via API |
| Tool calling | Manual prompt hacking | Native tool calling (Qwen3) |
| Docker complexity | Ollama container + volumes | No LLM container needed |
| Cold start | 30-60s model loading | Instant API response |
| Reliability | Depends on your hardware | 99.9% uptime serverless |

---

## 🛡️ Fallback Strategy

| Service | Fallback |
|---|---|
| Featherless Qwen3-32B | Switch to `Qwen/Qwen2.5-7B-Instruct` (faster) |
| Featherless Embeddings | Local `sentence-transformers/all-MiniLM-L6-v2` (dim=384) |
| Milvus | PostgreSQL + pgvector extension |
| Neo4j | NetworkX in-memory graph + JSON export |
| YOLOv8 + DeepSORT | YOLOv8n detections only (skip tracking) |
| Whisper | `faster-whisper` base model (already lightweight) |

**Priority if short on time:**

```
MUST HAVE (demo-critical):
  ✅ OCR + NLP summarization (Featherless)
  ✅ Investigation copilot chat (Featherless)
  ✅ Semantic search (Featherless embeddings + Milvus)

SHOULD HAVE (impressive):
  ✅ CCTV detection (local YOLO)
  ✅ Graph visualization (Neo4j)
  ✅ Risk scoring (pure Python)

NICE TO HAVE:
  ⬜ Tool-calling agents
  ⬜ Voice transcription
  ⬜ Legal report generation
```

---

## ✅ Final MVP Checklist

```
PHASE 1 — Foundation (Days 1-3)
  [ ] Python venv + requirements installed
  [ ] .env configured with Featherless API key
  [ ] Docker: Neo4j + Milvus + Redis running
  [ ] Featherless API verified (test chat completion)
  [ ] Featherless embeddings verified (test embedding generation)
  [ ] FastAPI skeleton running on :8001

PHASE 2 — Core Intelligence (Days 4-7)
  [ ] OCR extracting text from PDF/images
  [ ] NLP summarization via Featherless working
  [ ] Entity extraction (spaCy) working
  [ ] Embeddings stored in Milvus
  [ ] Semantic search returning results

PHASE 3 — CCTV + Graph (Days 8-10)
  [ ] YOLO detections on sample video
  [ ] CCTV events stored and retrievable
  [ ] Neo4j auto-populated from NER
  [ ] Graph API returning nodes/edges
  [ ] Mastermind detection working

PHASE 4 — Agents + Copilot (Days 11-13)
  [ ] Evidence agent running end-to-end
  [ ] Timeline agent running end-to-end
  [ ] Copilot responding in natural language
  [ ] Risk scoring returning scores + flags
  [ ] Legal report agent producing output

PHASE 5 — Integration + Polish (Days 14-15)
  [ ] Full demo flow tested end-to-end
  [ ] Integration with Member 1 verified
  [ ] API docs at /docs
  [ ] Error handling on all routes
  [ ] Sample case loaded for demo
  [ ] Demo script rehearsed
```

---

## ⚡ Quick Start

```bash
# Day 1 — Get everything running
git clone <repo>
cd backend

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate          # Windows
# source venv/bin/activate       # Linux/Mac

# Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# Start database services (NO Ollama needed!)
docker-compose up -d neo4j milvus redis

# Set your API key
echo "FEATHERLESS_API_KEY=rc_b6cb34a0e41abcd2266da87e740d1ec579d172e3412d675051b0bd1b73bac7c1" > .env

# Verify Featherless connection
python -c "
from openai import OpenAI
client = OpenAI(base_url='https://api.featherless.ai/v1', api_key='rc_b6cb34a0e41abcd2266da87e740d1ec579d172e3412d675051b0bd1b73bac7c1')
r = client.chat.completions.create(model='Qwen/Qwen2.5-7B-Instruct', messages=[{'role':'user','content':'Hello RAW'}])
print('✅ Featherless connected:', r.choices[0].message.content[:100])
"

# Start the API server
uvicorn app.main:app --reload --port 8001
# Visit http://localhost:8001/docs
```

---

*Master MVP Implementation Plan — RAW Investigation OS (Featherless AI Edition)*
*Version 2.0 | Hackathon Build*
