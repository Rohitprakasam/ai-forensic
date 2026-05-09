# Person 2 — AI Intelligence & OpenClaw Engineer
# Complete MVP Implementation Plan
# AI-Powered Unified Investigation Operating System (RAW)

---

## 📌 Role Summary

You are responsible for building the **entire AI intelligence layer** of the platform.
Member 1 handles the infrastructure. You consume their APIs and plug AI reasoning on top of everything.

Your deliverables are what make the platform impressive at demo time — the "wow factor."

---

## 🗂️ Table of Contents

1. [Technology Setup Checklist](#1-technology-setup-checklist)
2. [Folder Structure](#2-folder-structure)
3. [Week-by-Week Implementation Plan](#3-week-by-week-implementation-plan)
4. [Module 1 — OCR & NLP Pipeline](#4-module-1--ocr--nlp-pipeline)
5. [Module 2 — CCTV Analytics Service](#5-module-2--cctv-analytics-service)
6. [Module 3 — Graph Intelligence Service](#6-module-3--graph-intelligence-service)
7. [Module 4 — Semantic Search (Milvus)](#7-module-4--semantic-search-milvus)
8. [Module 5 — Risk Scoring Engine](#8-module-5--risk-scoring-engine)
9. [Module 6 — Voice Assistant Service](#9-module-6--voice-assistant-service)
10. [Module 7 — OpenClaw AI Agents](#10-module-7--openclaw-ai-agents)
11. [Module 8 — Investigation Copilot API](#11-module-8--investigation-copilot-api)
12. [Integration Points with Member 1](#12-integration-points-with-member-1)
13. [API Contracts You Expose](#13-api-contracts-you-expose)
14. [MVP Demo Flow](#14-mvp-demo-flow)
15. [Fallback & Simplification Strategy](#15-fallback--simplification-strategy)
16. [Final MVP Checklist](#16-final-mvp-checklist)

---

## 1. Technology Setup Checklist

Before writing any code, get your environment fully running.

### Python Environment

```bash
python -m venv venv
source venv/bin/activate
pip install fastapi uvicorn paddlepaddle paddleocr
pip install transformers sentence-transformers torch
pip install openai neo4j pymilvus redis celery
pip install opencv-python ultralytics deep-sort-realtime
pip install faster-whisper spacy pymupdf
python -m spacy download en_core_web_sm
```

### Docker Services (your responsibility)

```yaml
# Add to the shared docker-compose.yml

  neo4j:
    image: neo4j:5
    ports:
      - "7474:7474"
      - "7687:7687"
    environment:
      NEO4J_AUTH: neo4j/investigation123

  milvus:
    image: milvusdb/milvus:v2.4.0
    ports:
      - "19530:19530"

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
```

### Pull Local LLM (Llama 3 via Ollama)

```bash
docker exec -it ollama ollama pull llama3
# Fallback (lighter model):
docker exec -it ollama ollama pull mistral
```

---

## 2. Folder Structure

Your working area inside the shared `backend/` repo:

```
backend/
└── app/
    ├── ai/
    │   ├── openclaw/
    │   │   ├── __init__.py
    │   │   ├── agent_runner.py        # Core OpenClaw runtime
    │   │   ├── tool_registry.py       # All tools available to agents
    │   │   └── prompts/
    │   │       ├── evidence_prompt.txt
    │   │       ├── timeline_prompt.txt
    │   │       ├── autopsy_prompt.txt
    │   │       ├── legal_prompt.txt
    │   │       └── copilot_prompt.txt
    │   │
    │   ├── evidence_agent.py
    │   ├── timeline_agent.py
    │   ├── cctv_agent.py
    │   ├── autopsy_agent.py
    │   ├── graph_agent.py
    │   └── legal_report_agent.py
    │
    ├── services/
    │   ├── ocr_service.py
    │   ├── nlp_service.py
    │   ├── cctv_service.py
    │   ├── graph_service.py
    │   ├── search_service.py
    │   ├── risk_service.py
    │   └── voice_service.py
    │
    └── routes/
        ├── chatbot.py                 # Copilot endpoint
        ├── cctv.py
        └── voice.py
```

---

## 3. Week-by-Week Implementation Plan

```
WEEK 1  │ Environment + OCR/NLP + Embeddings Pipeline
WEEK 2  │ CCTV Analytics + Semantic Search (Milvus)
WEEK 3  │ Graph Intelligence (Neo4j) + Risk Scoring
WEEK 4  │ OpenClaw Agents + Investigation Copilot
WEEK 5  │ Integration + Polish + Demo Prep
```

---

## 4. Module 1 — OCR & NLP Pipeline

**Goal:** When Member 1 uploads a file (PDF, image, doc), your service extracts text, identifies entities, and generates an AI summary.

**Trigger:** Celery task fired after evidence upload by Member 1.

### `services/ocr_service.py`

```python
from paddleocr import PaddleOCR
import fitz  # PyMuPDF for PDFs

ocr = PaddleOCR(use_angle_cls=True, lang='en')

def extract_text_from_image(image_path: str) -> str:
    result = ocr.ocr(image_path, cls=True)
    lines = [word_info[1][0] for line in result for word_info in line]
    return "\n".join(lines)

def extract_text_from_pdf(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

def extract_text(file_path: str, file_type: str) -> str:
    if file_type in ["jpg", "jpeg", "png", "bmp"]:
        return extract_text_from_image(file_path)
    elif file_type == "pdf":
        return extract_text_from_pdf(file_path)
    return ""
```

### `services/nlp_service.py`

```python
import requests
import spacy

nlp = spacy.load("en_core_web_sm")

OLLAMA_URL = "http://ollama:11434/api/generate"

def summarize_text(text: str, context: str = "forensic evidence") -> str:
    prompt = f"""
    You are a forensic AI analyst. Analyze the following {context} document.
    Extract: key facts, persons mentioned, locations, dates, and suspicious elements.
    Provide a structured summary in 5-8 bullet points.

    DOCUMENT:
    {text[:3000]}

    SUMMARY:
    """
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })
    return response.json().get("response", "")

def extract_entities(text: str) -> dict:
    doc = nlp(text)
    return {
        "persons": list({ent.text for ent in doc.ents if ent.label_ == "PERSON"}),
        "locations": list({ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]}),
        "dates": list({ent.text for ent in doc.ents if ent.label_ == "DATE"}),
        "organizations": list({ent.text for ent in doc.ents if ent.label_ == "ORG"}),
        "money": list({ent.text for ent in doc.ents if ent.label_ == "MONEY"}),
    }

def generate_embeddings(text: str) -> list:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    return model.encode(text).tolist()
```

### Celery Task (in `workers/ai_tasks.py`)

```python
from app.services.ocr_service import extract_text
from app.services.nlp_service import summarize_text, extract_entities, generate_embeddings
from app.services.search_service import store_embedding

@celery_app.task
def process_evidence_ai(evidence_id: str, file_path: str, file_type: str):
    # Step 1: Extract text
    raw_text = extract_text(file_path, file_type)

    # Step 2: Summarize
    summary = summarize_text(raw_text)

    # Step 3: Extract entities
    entities = extract_entities(raw_text)

    # Step 4: Generate and store embeddings
    embedding = generate_embeddings(raw_text)
    store_embedding(evidence_id, embedding, metadata={"summary": summary})

    # Step 5: Save back to PostgreSQL (call Member 1's internal service)
    save_ai_results(evidence_id, summary, entities)

    # Step 6: Auto-populate graph
    from app.services.graph_service import add_entities_to_graph
    add_entities_to_graph(entities, evidence_id)
```

**API Endpoint:**

```python
# routes/upload.py (coordinate with Member 1 to fire this)
POST /ai/analyze/evidence
Body: { evidence_id, file_path, file_type }
```

---

## 5. Module 2 — CCTV Analytics Service

**Goal:** Accept a video file, detect persons and objects, track movement, produce a list of timestamped events.

### `services/cctv_service.py`

```python
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

model = YOLO("yolov8n.pt")  # Use yolov8n for speed; upgrade to yolov11 if available
tracker = DeepSort(max_age=30)

def analyze_video(video_path: str, case_id: str) -> list:
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    events = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO every 10th frame for performance
        if frame_count % 10 == 0:
            results = model(frame, verbose=False)
            detections = []

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    if conf > 0.4:
                        detections.append(([x1, y1, x2-x1, y2-y1], conf, label))

            tracks = tracker.update_tracks(detections, frame=frame)
            timestamp_sec = frame_count / fps

            for track in tracks:
                if track.is_confirmed():
                    events.append({
                        "timestamp": round(timestamp_sec, 2),
                        "track_id": track.track_id,
                        "class": track.det_class,
                        "bbox": track.to_ltrb().tolist(),
                        "case_id": case_id
                    })

        frame_count += 1

    cap.release()
    return events

def detect_suspicious_activity(events: list) -> list:
    """Flag events where the same person appears in multiple locations quickly."""
    flags = []
    from collections import defaultdict
    person_appearances = defaultdict(list)

    for event in events:
        if event["class"] == "person":
            person_appearances[event["track_id"]].append(event["timestamp"])

    for track_id, timestamps in person_appearances.items():
        if len(timestamps) > 20:  # Loitering detection
            flags.append({
                "type": "loitering",
                "track_id": track_id,
                "duration_seconds": max(timestamps) - min(timestamps),
                "severity": "medium"
            })

    return flags
```

### `routes/cctv.py`

```python
from fastapi import APIRouter, UploadFile, File, BackgroundTasks
from app.services.cctv_service import analyze_video, detect_suspicious_activity
import shutil, uuid

router = APIRouter(prefix="/cctv", tags=["CCTV Analytics"])

@router.post("/analyze/{case_id}")
async def analyze_cctv(case_id: str, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    video_id = str(uuid.uuid4())
    save_path = f"/tmp/{video_id}.mp4"

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    background_tasks.add_task(run_cctv_pipeline, case_id, save_path, video_id)
    return {"status": "processing", "video_id": video_id}

@router.get("/events/{case_id}")
async def get_cctv_events(case_id: str):
    # Return stored events from PostgreSQL
    return fetch_cctv_events(case_id)

def run_cctv_pipeline(case_id, path, video_id):
    events = analyze_video(path, case_id)
    flags = detect_suspicious_activity(events)
    store_cctv_events(case_id, events, flags)
```

---

## 6. Module 3 — Graph Intelligence Service

**Goal:** Build a relationship graph linking suspects, victims, locations, and evidence. Detect hidden connections.

### Neo4j Schema

```cypher
// Nodes
CREATE (p:Person {name: "John Doe", role: "suspect"})
CREATE (l:Location {name: "Market Street", type: "crime_scene"})
CREATE (e:Evidence {id: "ev_001", type: "document"})
CREATE (c:Case {id: "case_001", title: "Murder Case"})

// Relationships
(p)-[:SEEN_AT]->(l)
(p)-[:LINKED_TO]->(e)
(p)-[:INVOLVED_IN]->(c)
(p)-[:KNOWS]->(p2)
(l)-[:PART_OF]->(c)
```

### `services/graph_service.py`

```python
from neo4j import GraphDatabase

driver = GraphDatabase.driver("bolt://neo4j:7687", auth=("neo4j", "investigation123"))

def add_entities_to_graph(entities: dict, evidence_id: str):
    with driver.session() as session:
        for person in entities.get("persons", []):
            session.run(
                "MERGE (p:Person {name: $name}) "
                "MERGE (e:Evidence {id: $eid}) "
                "MERGE (p)-[:LINKED_TO]->(e)",
                name=person, eid=evidence_id
            )
        for location in entities.get("locations", []):
            session.run(
                "MERGE (l:Location {name: $name}) "
                "MERGE (e:Evidence {id: $eid}) "
                "MERGE (l)-[:MENTIONED_IN]->(e)",
                name=location, eid=evidence_id
            )

def find_connected_suspects(person_name: str) -> list:
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person {name: $name})-[*1..3]-(connected)
            RETURN connected, labels(connected) as type
            LIMIT 50
            """,
            name=person_name
        )
        return [{"node": dict(r["connected"]), "type": r["type"]} for r in result]

def get_full_case_graph(case_id: str) -> dict:
    with driver.session() as session:
        nodes_result = session.run(
            "MATCH (n)-[:PART_OF|INVOLVED_IN|LINKED_TO*0..2]->(c:Case {id: $cid}) "
            "RETURN n, labels(n) as type LIMIT 100",
            cid=case_id
        )
        edges_result = session.run(
            "MATCH (a)-[r]->(b) WHERE (a)-[:INVOLVED_IN*0..3]->(:Case {id: $cid}) "
            "RETURN a.name as source, type(r) as rel, b.name as target LIMIT 200",
            cid=case_id
        )
        return {
            "nodes": [{"id": dict(r["n"]).get("name",""), "type": r["type"][0]} for r in nodes_result],
            "edges": [{"source": r["source"], "relation": r["rel"], "target": r["target"]} for r in edges_result]
        }

def detect_mastermind(case_id: str) -> list:
    """Find the most connected person — likely the organizer."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (p:Person)-[:INVOLVED_IN]->(:Case {id: $cid})
            WITH p, size((p)--()) as connections
            ORDER BY connections DESC
            RETURN p.name as name, connections
            LIMIT 5
            """,
            cid=case_id
        )
        return [{"suspect": r["name"], "connection_count": r["connections"]} for r in result]
```

### API

```python
GET  /graph/case/{case_id}          → full graph JSON for visualization
GET  /graph/suspect/{name}          → connections of a specific person
GET  /graph/mastermind/{case_id}    → most connected suspects
POST /graph/add-relationship        → manually add a relationship
```

---

## 7. Module 4 — Semantic Search (Milvus)

**Goal:** Store evidence embeddings so investigators can search by meaning, not just keywords. "Show me all evidence about financial transactions before the incident."

### `services/search_service.py`

```python
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility

connections.connect("default", host="milvus", port="19530")

COLLECTION_NAME = "evidence_embeddings"

def init_milvus():
    if not utility.has_collection(COLLECTION_NAME):
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="evidence_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="case_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=384),
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
    collection.insert([[evidence_id], [metadata.get("case_id", "")],
                        [metadata.get("summary", "")], [embedding]])
    collection.flush()

def semantic_search(query: str, case_id: str = None, top_k: int = 5) -> list:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode(query).tolist()

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

    hits = []
    for result in results[0]:
        hits.append({
            "evidence_id": result.entity.get("evidence_id"),
            "summary": result.entity.get("summary"),
            "relevance_score": round(result.score, 4)
        })
    return hits
```

### API

```python
POST /search/semantic
Body: { query: "suspicious money transfers", case_id: "case_001", top_k: 5 }

GET /search/keyword?q=knife&case_id=case_001
```

---

## 8. Module 5 — Risk Scoring Engine

**Goal:** Given all evidence for a case, generate a risk score and flag anomalies.

### `services/risk_service.py`

```python
def calculate_risk_score(case_data: dict) -> dict:
    score = 0
    flags = []

    evidence_list = case_data.get("evidence", [])
    timeline_events = case_data.get("timeline_events", [])
    graph_data = case_data.get("graph", {})

    # Rule 1: Volume of evidence
    if len(evidence_list) > 20:
        score += 15
        flags.append("High evidence volume — complex case")

    # Rule 2: Multiple suspects connected
    suspects = [n for n in graph_data.get("nodes", []) if n.get("type") == "Person"]
    if len(suspects) > 3:
        score += 20
        flags.append(f"{len(suspects)} connected individuals detected")

    # Rule 3: Financial evidence
    financial_keywords = ["transfer", "payment", "cash", "bank", "transaction"]
    for ev in evidence_list:
        summary = ev.get("ai_summary", "").lower()
        if any(kw in summary for kw in financial_keywords):
            score += 10
            flags.append("Financial evidence detected")
            break

    # Rule 4: Tight timeline clustering
    if len(timeline_events) > 10:
        timestamps = sorted([e["timestamp"] for e in timeline_events])
        gaps = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        if min(gaps) < 300:  # Events within 5 minutes
            score += 15
            flags.append("Rapid sequence of events detected")

    # Rule 5: Weapon/violence keywords
    violence_keywords = ["weapon", "knife", "gun", "threat", "attack", "assault"]
    for ev in evidence_list:
        summary = ev.get("ai_summary", "").lower()
        if any(kw in summary for kw in violence_keywords):
            score += 25
            flags.append("Weapon or violence evidence present")
            break

    risk_level = "LOW" if score < 30 else "MEDIUM" if score < 60 else "HIGH" if score < 80 else "CRITICAL"

    return {
        "score": min(score, 100),
        "risk_level": risk_level,
        "flags": list(set(flags)),
        "recommendation": get_recommendation(risk_level)
    }

def get_recommendation(level: str) -> str:
    recommendations = {
        "LOW": "Standard investigation protocols apply.",
        "MEDIUM": "Prioritize evidence review. Additional officers recommended.",
        "HIGH": "Immediate senior review required. Consider arrest warrants.",
        "CRITICAL": "URGENT — Escalate immediately. Full task force deployment recommended."
    }
    return recommendations.get(level, "")
```

---

## 9. Module 6 — Voice Assistant Service

**Goal:** Transcribe audio recordings (interrogations, witness statements), extract entities.

### `services/voice_service.py`

```python
from faster_whisper import WhisperModel

model = WhisperModel("base", device="cpu", compute_type="int8")

def transcribe_audio(audio_path: str) -> dict:
    segments, info = model.transcribe(audio_path, beam_size=5)

    full_text = ""
    timestamped = []

    for segment in segments:
        full_text += segment.text + " "
        timestamped.append({
            "start": round(segment.start, 2),
            "end": round(segment.end, 2),
            "text": segment.text.strip()
        })

    return {
        "full_transcript": full_text.strip(),
        "language": info.language,
        "segments": timestamped
    }
```

### `routes/voice.py`

```python
from fastapi import APIRouter, UploadFile, File
from app.services.voice_service import transcribe_audio
from app.services.nlp_service import extract_entities, summarize_text
import shutil, uuid

router = APIRouter(prefix="/voice", tags=["Voice Intelligence"])

@router.post("/transcribe/{case_id}")
async def transcribe(case_id: str, file: UploadFile = File(...)):
    audio_id = str(uuid.uuid4())
    path = f"/tmp/{audio_id}.wav"

    with open(path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    result = transcribe_audio(path)
    entities = extract_entities(result["full_transcript"])
    summary = summarize_text(result["full_transcript"], context="witness statement / interrogation")

    return {
        "transcript": result,
        "entities": entities,
        "ai_summary": summary,
        "case_id": case_id
    }
```

---

## 10. Module 7 — OpenClaw AI Agents

**Goal:** Autonomous agents that reason over case data and produce investigation intelligence.

### Agent Architecture

Each agent follows the pattern:
1. Receive case context
2. Call the relevant tools/services
3. Generate a structured AI analysis
4. Return results to be stored

### `ai/openclaw/agent_runner.py`

```python
import requests

OLLAMA_URL = "http://ollama:11434/api/generate"

def run_agent(agent_name: str, system_prompt: str, context: str) -> str:
    """Core agent runner — sends context + system prompt to LLM."""
    full_prompt = f"""
{system_prompt}

CASE CONTEXT:
{context}

Provide your structured forensic analysis below. Be specific, factual, and actionable.
"""
    response = requests.post(OLLAMA_URL, json={
        "model": "llama3",
        "prompt": full_prompt,
        "stream": False
    })
    return response.json().get("response", "Agent did not return a response.")
```

### `ai/evidence_agent.py`

```python
from app.ai.openclaw.agent_runner import run_agent

EVIDENCE_SYSTEM_PROMPT = """
You are a forensic evidence analysis AI agent.
Your job is to analyze collected evidence and identify:
- Key facts established by the evidence
- Inconsistencies or contradictions
- Missing evidence that should be collected
- Connections between pieces of evidence
- Priority evidence for prosecution

Always structure your output with clear sections.
Never speculate beyond what the evidence supports.
"""

def analyze_evidence(case_id: str, evidence_summaries: list) -> str:
    context = f"Case ID: {case_id}\n\nEvidence Summaries:\n"
    for i, ev in enumerate(evidence_summaries, 1):
        context += f"\n[{i}] Type: {ev['type']}\nSummary: {ev['summary']}\n"

    return run_agent("evidence_agent", EVIDENCE_SYSTEM_PROMPT, context)
```

### `ai/timeline_agent.py`

```python
from app.ai.openclaw.agent_runner import run_agent

TIMELINE_SYSTEM_PROMPT = """
You are a forensic timeline reconstruction AI agent.
Given a list of events, your job is to:
- Reconstruct the chronological sequence of the incident
- Identify gaps in the timeline where information is missing
- Detect anomalies (events that don't fit the pattern)
- Suggest what likely happened during unknown periods
- Calculate time between critical events

Always present the timeline in chronological order with timestamps.
"""

def reconstruct_timeline(case_id: str, events: list) -> str:
    context = f"Case ID: {case_id}\n\nRaw Events:\n"
    for event in sorted(events, key=lambda x: x.get("timestamp", 0)):
        context += f"- [{event['timestamp']}] {event['event_type']}: {event.get('description', '')}\n"

    return run_agent("timeline_agent", TIMELINE_SYSTEM_PROMPT, context)
```

### `ai/legal_report_agent.py`

```python
from app.ai.openclaw.agent_runner import run_agent

LEGAL_REPORT_PROMPT = """
You are a forensic legal report generation AI agent.
Generate a structured investigation report with the following sections:
1. Case Summary
2. Evidence Overview
3. Suspect Analysis
4. Timeline of Events
5. Key Findings
6. Recommended Next Steps
7. Confidence Assessment (Low / Medium / High)

The report must be suitable for submission to a court or senior law enforcement officer.
Use formal, precise language. Do not speculate. Cite evidence.
"""

def generate_legal_report(case_data: dict) -> str:
    context = f"""
Case ID: {case_data['case_id']}
Crime Type: {case_data['crime_type']}
Suspects: {', '.join(case_data.get('suspects', []))}
Evidence Count: {len(case_data.get('evidence', []))}
Risk Score: {case_data.get('risk_score', 'N/A')}
Timeline Events: {len(case_data.get('timeline_events', []))}
AI Summaries: {' | '.join([e.get('summary','') for e in case_data.get('evidence', [])][:5])}
"""
    return run_agent("legal_agent", LEGAL_REPORT_PROMPT, context)
```

---

## 11. Module 8 — Investigation Copilot API

**Goal:** A conversational AI that investigators can query in natural language to get intelligence about an active case.

### `routes/chatbot.py`

```python
from fastapi import APIRouter
from pydantic import BaseModel
import requests

router = APIRouter(prefix="/copilot", tags=["Investigation Copilot"])

OLLAMA_URL = "http://ollama:11434/api/generate"

class CopilotQuery(BaseModel):
    case_id: str
    question: str
    conversation_history: list = []

@router.post("/ask")
async def ask_copilot(query: CopilotQuery):
    # Fetch live case context
    case_context = build_case_context(query.case_id)

    # Build conversation string from history
    history_str = ""
    for msg in query.conversation_history[-6:]:  # last 3 turns
        history_str += f"Investigator: {msg['question']}\nCopilot: {msg['answer']}\n\n"

    prompt = f"""
You are RAW — an AI-powered forensic investigation copilot.
You have full access to the case intelligence below.
Answer the investigator's question concisely and accurately.
If you do not have enough data to answer, say so honestly.
Never fabricate facts.

CASE INTELLIGENCE:
{case_context}

PREVIOUS CONVERSATION:
{history_str}

INVESTIGATOR'S QUESTION: {query.question}

COPILOT RESPONSE:
"""

    response = requests.post(OLLAMA_URL, json={
        "model": "llama3",
        "prompt": prompt,
        "stream": False
    })

    answer = response.json().get("response", "")
    return {"answer": answer, "case_id": query.case_id}


def build_case_context(case_id: str) -> str:
    """Aggregate all intelligence for a case into a context string."""
    # This calls Member 1's APIs internally
    import httpx
    base = "http://localhost:8000"

    case = httpx.get(f"{base}/cases/{case_id}").json()
    evidence = httpx.get(f"{base}/evidence/case/{case_id}").json()
    timeline = httpx.get(f"{base}/timeline/{case_id}").json()

    context = f"Case: {case.get('title')} | Crime: {case.get('crime_type')} | Status: {case.get('status')}\n"
    context += f"Evidence items: {len(evidence)}\n"
    context += f"Timeline events: {len(timeline)}\n"

    for ev in evidence[:5]:
        context += f"- Evidence: {ev.get('type')} | AI Summary: {ev.get('ai_summary', 'Not analyzed')}\n"

    return context
```

---

## 12. Integration Points with Member 1

This is how your AI services plug into Member 1's infrastructure.

| What You Need | How You Get It |
|---|---|
| Evidence file path after upload | Member 1 fires a Celery task → your `process_evidence_ai()` |
| Case data for agents | Call `GET /cases/:id` |
| Store AI results (summary, entities) | Ask Member 1 to add `ai_summary`, `entities` columns to `evidence` table |
| Store CCTV events | Ask Member 1 to add a `cctv_events` table |
| Store risk scores | Ask Member 1 to add `risk_score` to `cases` table |
| Timeline events | Call `POST /timeline/event` after CCTV/NLP processing |

### Shared Task Queue Contract

Member 1 should fire this Celery task after every upload:

```python
# Member 1 fires this after evidence upload:
process_evidence_ai.delay(
    evidence_id=str(evidence.id),
    file_path=minio_path,
    file_type=file_extension
)
```

---

## 13. API Contracts You Expose

All routes you own, to be consumed by Member 1's gateway and the frontend:

```
POST   /ai/analyze/evidence             → Trigger AI analysis on an evidence item
POST   /cctv/analyze/{case_id}          → Upload + analyze CCTV footage
GET    /cctv/events/{case_id}           → Get all CCTV events for a case
GET    /graph/case/{case_id}            → Get full relationship graph
GET    /graph/suspect/{name}            → Get suspect connections
GET    /graph/mastermind/{case_id}      → Get most connected suspects
POST   /search/semantic                 → Semantic search across evidence
GET    /search/keyword                  → Keyword search
POST   /voice/transcribe/{case_id}      → Transcribe audio file
POST   /agents/evidence/{case_id}       → Run evidence agent
POST   /agents/timeline/{case_id}       → Run timeline agent
POST   /agents/legal-report/{case_id}   → Generate legal report
GET    /risk/score/{case_id}            → Get risk score for a case
POST   /copilot/ask                     → Investigation copilot chat
```

---

## 14. MVP Demo Flow

This is the exact flow you should be able to demonstrate live:

```
1. Investigator uploads a PDF evidence file
         ↓
2. OCR extracts text → NLP summarizes → entities extracted
         ↓
3. Embedding stored in Milvus
         ↓
4. Persons/locations auto-added to Neo4j graph
         ↓
5. Investigator uploads CCTV footage
         ↓
6. YOLO detects persons → DeepSORT tracks movement → events logged
         ↓
7. Investigator types: "Who are the connected suspects?"
   → Graph Intelligence returns suspect network
         ↓
8. Investigator types in copilot: "What happened between 9pm and 11pm?"
   → Copilot answers using timeline + evidence context
         ↓
9. Semantic search: "Show me all financial evidence"
   → Milvus returns top relevant documents
         ↓
10. Risk Score calculated → Case flagged as HIGH
         ↓
11. Legal Report Agent generates full investigation report
         ↓
12. DEMO COMPLETE ✅
```

---

## 15. Fallback & Simplification Strategy

If time runs out or a service has issues, use these fallbacks:

| Service | Fallback |
|---|---|
| Llama 3 (local) | Use OpenAI API (GPT-4o-mini) with env key |
| Milvus | Use PostgreSQL vector extension (pgvector) |
| Neo4j | Use NetworkX in-memory graph + JSON export |
| YOLOv11 | Use YOLOv8n (pre-trained, faster to set up) |
| DeepSORT | Skip tracking, just return YOLO detection events |
| Whisper (full) | Use `faster-whisper` base model only |
| Full agent pipeline | Hardcode one impressive demo agent for presentation |

**Priority order if short on time:**

```
MUST HAVE (demo-critical):
  ✅ OCR + NLP summarization
  ✅ Investigation copilot (chat)
  ✅ Semantic search

SHOULD HAVE (impressive):
  ✅ CCTV detection (basic YOLO, no tracking)
  ✅ Graph visualization
  ✅ Risk scoring

NICE TO HAVE (if time allows):
  ⬜ DeepSORT tracking
  ⬜ Voice transcription
  ⬜ Legal report generation
  ⬜ Autopsy agent
```

---

## 16. Final MVP Checklist

```
WEEK 1
  [ ] Docker services running: Neo4j, Milvus, Ollama
  [ ] Llama 3 pulled and responding
  [ ] OCR working on a sample PDF
  [ ] NLP summarization working
  [ ] Embeddings generating correctly
  [ ] Milvus storing and retrieving embeddings

WEEK 2
  [ ] CCTV pipeline: YOLO detections working
  [ ] CCTV events being stored
  [ ] Semantic search returning relevant results
  [ ] Keyword search working

WEEK 3
  [ ] Neo4j auto-populated from NLP entity extraction
  [ ] Graph API returning node/edge data
  [ ] Mastermind detection working
  [ ] Risk scoring returning scores + flags

WEEK 4
  [ ] Evidence agent running end-to-end
  [ ] Timeline agent running end-to-end
  [ ] Legal report agent producing output
  [ ] Copilot API responding in natural language
  [ ] All routes tested with Postman / curl

WEEK 5
  [ ] Full demo flow tested end-to-end
  [ ] Integration with Member 1 verified
  [ ] API docs generated (FastAPI /docs)
  [ ] Error handling on all routes
  [ ] At least 1 sample case loaded with evidence for demo
  [ ] Demo script rehearsed
```

---

## ⚡ Quick Start Command

```bash
# Day 1 — get everything running
git clone <repo>
cd backend
docker-compose up -d neo4j milvus ollama redis postgres
docker exec -it ollama ollama pull llama3
source venv/bin/activate
uvicorn app.main:app --reload --port 8001
# Visit http://localhost:8001/docs
```

---

*Person 2 Implementation Plan — AI-Powered Unified Investigation OS (RAW)*
*Version 1.0 | Hackathon Build*
