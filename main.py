# ─────────────────────────────────────────────
# 1. Imports & Path Setup
# ─────────────────────────────────────────────
# Standard library imports for handling paths, system modules, and logging
import sys
import logging
import re
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

# FastAPI is our web framework for building the API
from fastapi import FastAPI, HTTPException

# Pydantic is used for data validation and defining the structure of our JSON requests/responses
from pydantic import BaseModel

# ─────────────────────────────────────────────
# 2. Shared Module Imports
# ─────────────────────────────────────────────
# We add "Program_Ingest" to the Python path so we can import modules from that folder
# This lets main.py use the logic defined in ingest.py and shared.py
sys.path.append(str(Path(__file__).parent / "Program_Ingest"))

from Program_Ingest.shared import load_tree, KB_DIR, KB_FILE, TreeNode, create_node_map

# ─────────────────────────────────────────────
# 3. External Dependencies & Fallbacks
# ─────────────────────────────────────────────
# Try importing ChatOllama (part of LangChain), which connects to our local Ollama server
try:
    from langchain_ollama import ChatOllama
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False
    # If LangChain isn't installed, we'll disable the AI "Librarian" features
    print("WARNING: langchain_ollama not found. AI features will be disabled.")

# ─────────────────────────────────────────────
# 4. Initialization & Global State
# ─────────────────────────────────────────────
# These variables hold the "brain" of the application once it's loaded from disk
tree = None           # The hierarchical knowledge index
node_map = {}         # A flat map of {id: node} for fast lookups
librarian_model = None # The AI model (Ministral-3b)
holocron_status = "Initializing..."
IS_LIBRARIAN_READY = False

# ─────────────────────────────────────────────
# 5. Configuration
# ─────────────────────────────────────────────
# Change this to use a different model from Ollama (e.g., "gemma3:1b")
LIBRARIAN_MODEL_NAME = "tinyllama:latest"

# ─────────────────────────────────────────────
# 6. API Lifecycle (Lifespan)
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    This function runs once when the server starts and once when it stops.
    It's the perfect place to load the knowledge base and warm up the AI.
    """
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

# Initialize the FastAPI app with our custom startup/shutdown logic
app = FastAPI(lifespan=lifespan)

# ─────────────────────────────────────────────
# 7. Request & Response Models (Pydantic)
# ─────────────────────────────────────────────
# These classes define what valid data looks like for our API endpoints.
# If a user sends data that doesn't match these, FastAPI returns a 422 error automatically.

class NodeRequest(BaseModel):
    """User sends a node ID to retrieve its content."""
    node_id: str

class NodeResponse(BaseModel):
    """The API returns detailed info about a specific node."""
    node_id: str
    title: str
    content: str
    domain: str

class SearchRequest(BaseModel):
    """User sends a search query and an optional results limit."""
    query: str
    limit: int = 5

class SearchResponse(BaseModel):
    """The API returns a list of matching nodes."""
    results: list[NodeResponse]

class AskRequest(BaseModel):
    """User asks a question using the full RAG pipeline."""
    query: str
    context_limit: int = 3

class AskResponse(BaseModel):
    """The AI's answer along with the source materials used."""
    answer: str
    sources: list[str]

# ─────────────────────────────────────────────
# 8. API Endpoints
# ─────────────────────────────────────────────

@app.get("/health")
def health():
    """Check if the API and the AI Librarian are online."""
    return {
        "status": "online",
        "kb_nodes": tree.node_count() if tree else 0,
        "librarian_status": holocron_status,
        "model": LIBRARIAN_MODEL_NAME,
        "ready": IS_LIBRARIAN_READY
    }

@app.get("/tree")
def get_tree_structure():
    """Return the full tree skeleton (titles only, no content) for UI display."""
    if not tree:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    return tree.to_dict(include_content=False)

@app.post("/node", response_model=NodeResponse)
def get_node(req: NodeRequest):
    """Retrieve full content for a specific node ID by looking it up in the map."""
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
    Keyword search for nodes. 
    It splits the user query into terms and compares them against node titles and content.
    A simple scoring system prioritizes exact phrase matches in titles.
    """
    if not tree:
        raise HTTPException(status_code=503, detail="Knowledge base not loaded")
    
    results = []
    q_raw = req.query.lower()
    
    # Split query into terms (ignoring very short words like 'the', 'is')
    terms = [t for t in re.split(r'\W+', q_raw) if len(t) > 2]
    
    matches = []
    for nid, node in node_map.items():
        score = 0
        node_title = node.title.lower()
        node_content = node.content.lower()
        
        # Exact Phrase Match (Highest points)
        if q_raw in node_title:
            score += 50
        if q_raw in node_content:
            score += 20
            
        # Individual Term Match (Additive points)
        for term in terms:
            if term in node_title:
                score += 10
            if term in node_content:
                score += 1
                
        if score > 0:
            matches.append((score, node))
            
    # Sort matches by score (highest first)
    matches.sort(key=lambda x: x[0], reverse=True)
    
    # Extract the top results up to the requested 'limit'
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
    The Intelligent Librarian RAG Endpoint.
    1. Search: Finds the most relevant notes in the Holocron.
    2. Context: Packs those notes into a prompt for the AI.
    3. Generation: The Librarian synthesizes a human-readable answer.
    """
    if not IS_LIBRARIAN_READY or not librarian_model:
        raise HTTPException(status_code=503, detail=f"Librarian model not ready. Status: {holocron_status}")
    
    # ── Step 1: Retrieval ──
    # We call our own search logic to find notes based on the user's question
    search_res = search_nodes(SearchRequest(query=req.query, limit=req.context_limit))
    nodes = search_res.results
    
    if not nodes:
        # If we can't find anything relevant in our notes, we should be honest
        return AskResponse(answer="I scanned the Holocron but could not find any specific notes matching your query.", sources=[])
    
    # ── Step 2: Context Construction ──
    # Flatten the retrieved notes into a single block of text for the AI to read
    context_text = "\n\n".join([f"--- Note: {n.title} ---\n{n.content}" for n in nodes])
    sources = [n.title for n in nodes]
    
    # ── Step 3: Generation (AI Reasoning) ──
    # We give the AI a very specific role (Librarian) and instructions to be factual
    prompt = (
        f"You are the Librarian. Use the following notes from the Holocron to answer the user's question.\n"
        f"Keep your answer focused and factual. If the answer is not in the notes, say so.\n\n"
        f"Context:\n{context_text}\n\n"
        f"User Question: {req.query}"
    )
    
    try:
        # Ask the model to generate the answer
        response = librarian_model.invoke(prompt)
        return AskResponse(answer=response.content, sources=sources)
    except Exception as e:
        # If the AI crashes or times out, report it as a 500 error
        raise HTTPException(status_code=500, detail=f"Librarian Generation Failed: {e}")
