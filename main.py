from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import sys
from pathlib import Path
import logging
from typing import List, Optional

# Add Program_Ingest to path to import shared module
sys.path.append(str(Path(__file__).parent / "Program_Ingest"))

from Program_Ingest.shared import load_tree, KB_DIR, KB_FILE, TreeNode, create_node_map

# Try importing ChatOllama, fallback to subprocess/requests if missing
try:
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # We will implement a simple fallback if needed, but for now assume it's there or user installs it
    # given it was in ingest.py

import re

# Global State
tree = None
node_map = {}
librarian_model = None
holocron_status = "Initializing..."
IS_LIBRARIAN_READY = False

# Configuration
LIBRARIAN_MODEL_NAME = "tinyllama:latest"

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tree, node_map, librarian_model, holocron_status, IS_LIBRARIAN_READY
    
    # 1. Load Knowledge Base
    kb_path = KB_FILE
    if kb_path.exists():
        tree = load_tree(kb_path)
        node_map = create_node_map(tree)
        print(f"Loaded knowledge base: {tree.node_count()} nodes")
    else:
        print("Knowledge base not found. Please run ingest.py to build it.")
        holocron_status = "Knowledge Base Missing."
        yield
        return

    # 2. Initialize Librarian (Ministral-3b)
    if HAS_LANGCHAIN:
        try:
            print(f"Initializing Librarian ({LIBRARIAN_MODEL_NAME})...")
            librarian_model = ChatOllama(model=LIBRARIAN_MODEL_NAME)
            
            # 3. Startup Scan / Warmup
            print("Librarian: Scanning Holocron Index...")
            # Create a summary of the tree root
            root_summary = ", ".join([n.title for n in tree.nodes[:20]]) # First 20 root nodes
            prompt = f"System Boot. You are the Librarian. The Holocron contains: {root_summary}. Briefly confirm you are ready."
            
            # Warmup invocation
            response = librarian_model.invoke(prompt)
            holocron_status = f"Online. {response.content}"
            print(f"Librarian Status: {holocron_status}")
            IS_LIBRARIAN_READY = True
            
        except Exception as e:
            print(f"Failed to initialize Librarian: {e}")
            holocron_status = f"Librarian Offline: {e}"
            IS_LIBRARIAN_READY = False
    else:
        print("LangChain not found. Librarian features disabled.")
        holocron_status = "Librarian Module Missing (LangChain)."
        IS_LIBRARIAN_READY = False

    yield

app = FastAPI(lifespan=lifespan)

# ─────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────
class NodeRequest(BaseModel):
    node_id: str

class NodeResponse(BaseModel):
    node_id: str
    title: str
    content: str
    domain: str

class SearchRequest(BaseModel):
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    results: list[NodeResponse]

class AskRequest(BaseModel):
    query: str
    context_limit: int = 3

class AskResponse(BaseModel):
    answer: str
    sources: list[str]

# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "online",
        "kb_nodes": tree.node_count() if tree else 0,
        "librarian_status": holocron_status,
        "model": LIBRARIAN_MODEL_NAME,
        "ready": IS_LIBRARIAN_READY
    }

@app.get("/tree")
def get_tree_structure():
    """Return the full tree skeleton (titles only, no content)."""
    if not tree:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    return tree.to_dict(include_content=False)

@app.post("/node", response_model=NodeResponse)
def get_node(req: NodeRequest):
    """Retrieve full content for a specific node ID."""
    if not tree:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    if req.node_id in node_map:
        n = node_map[req.node_id]
        return NodeResponse(
            node_id=n.node_id,
            title=n.title,
            content=n.content,
            domain=n.domain
        )
    raise HTTPException(status_code=404, detail="Node not found")

@app.post("/search", response_model=SearchResponse)
def search_nodes(req: SearchRequest):
    """
    Keyword search for nodes. Handles natural language via term splitting.
    """
    if not tree:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    results = []
    q_raw = req.query.lower()
    
    # Split into terms (simple tokenization)
    terms = [t for t in re.split(r'\W+', q_raw) if len(t) > 2] # Ignore words <= 2 chars
    
    matches = []
    for nid, node in node_map.items():
        score = 0
        node_title = node.title.lower()
        node_content = node.content.lower()
        
        # Exact Phrase Match (Highest Priority)
        if q_raw in node_title:
            score += 50
        if q_raw in node_content:
            score += 20
            
        # Term Match (Additive)
        for term in terms:
            if term in node_title:
                score += 10
            if term in node_content:
                score += 1
                
        if score > 0:
            matches.append((score, node))
            
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate by content roughly? No need for now.
    
    for _, node in matches[:req.limit]:
        results.append(NodeResponse(
            node_id=node.node_id,
            title=node.title,
            content=node.content,
            domain=node.domain
        ))
        
    return SearchResponse(results=results)

@app.post("/ask", response_model=AskResponse)
def ask_librarian(req: AskRequest):
    """
    Intelligent RAG Endpoint.
    1. Search for relevant nodes.
    2. Synthesize answer using Ministral-3b.
    """
    if not IS_LIBRARIAN_READY or not librarian_model:
        raise HTTPException(status_code=503, detail=f"Librarian model not ready. Status: {holocron_status}")
    
    # 1. Retrieval
    # We use our own search endpoint logic or call it directly
    search_res = search_nodes(SearchRequest(query=req.query, limit=req.context_limit))
    nodes = search_res.results
    
    if not nodes:
        # Fallback: Ask general knowledge if no nodes found? 
        # Or just be honest.
        # User requested "search notes". 
        return AskResponse(answer="I scanned the Holocron but could not find any specific notes matching your query.", sources=[])
    
    # 2. Context Construction
    context_text = "\n\n".join([f"--- Note: {n.title} ---\n{n.content}" for n in nodes])
    sources = [n.title for n in nodes]
    
    # 3. Generation
    prompt = (
        f"You are the Librarian. Use the following notes from the Holocron to answer the user's question.\n"
        f"Keep your answer focused and factual. If the answer is not in the notes, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Question: {req.query}"
    )
    
    try:
        response = librarian_model.invoke(prompt)
        return AskResponse(answer=response.content, sources=sources)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Librarian Generation Failed: {e}")
