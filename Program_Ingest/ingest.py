# Standard Library Imports
import json       # For reading/writing JSON files (saving the tree/manifest)
import re         # Regular expressions for text processing (finding headers)
import psutil     # For monitoring system memory usage to prevent crashes
import os         # Operating system interfaces (finding file paths)
import time       # For tracking how long operations take

# Data Structure Helpers
from dataclasses import dataclass, field  # Easy class definitions for the Tree
from pathlib import Path                  # Object-oriented filesystem paths
from typing import List, Dict, Tuple      # Type hinting for better code clarity

# External Libraries
import tiktoken       # OpenAI's tokenizer (used to count tokens accurately)
import pymupdf4llm    # Converts PDF layers to clean Markdown for LLMs

# AI & UI Libraries
from langchain_ollama import ChatOllama   # Connects to the local Ollama server
from pydantic import BaseModel, Field     # Data validation for structured outputs
from langchain_text_splitters import MarkdownHeaderTextSplitter # Splits MD by headers
from rich.console import Console          # Pretty printing to terminal
from rich.panel import Panel              # Boxed text in terminal
from rich.syntax import Syntax            # Syntax highlighting for JSON
from rich.tree import Tree as RichTree    # Visual tree structure in terminal
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn # Progress bars

##  ##                                                                                            ##  ##  --  --  Paths  --  --  ##  ##
BASE_DIR = Path.home() / "docker/APIinterface/Holonet"
KB_DIR = BASE_DIR / "Knowledgebase_Alpha"
LIBRARY_DIR = KB_DIR / "Library"
MD_CACHE_DIR = KB_DIR / "markdown_cache"
KB_FILE = KB_DIR / "holonet_resources.json"
MANIFEST_FILE = KB_DIR / "processed_files.json"

##  ##                                                                                            ##  ##  --  --  Tuning Constants  --  --  ##  ##
MAX_NODES_DISPLAY = 5           # Limit how many nodes we show in the summary
MAX_CONTEXT_TOKENS = 4500       # Maximum size of text fed to the LLM for answering
MAX_RETRIEVAL_NODES = 4         # How many chunks to retrieve for an answer
MAX_SUMMARY_TOKENS = 3000       # Truncate content before summarizing to save time
FALLBACK_CHUNK_TOKENS = 1500    # If headers are missing, chop into 1500-token blocks
MIN_HEADER_COUNT = 2            # If a PDF has < 2 headers, assume formatting failed

console = Console()
KB_DIR.mkdir(parents=True, exist_ok=True)
LIBRARY_DIR.mkdir(parents=True, exist_ok=True)
MD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

##  ##                                                                                            ##  ##  --  --  Model  --  --  ##  ##
model = ChatOllama(
    model="gemma3:1b",  # Using a small, fast model for local inference
    temperature=0,      # temperature=0 makes it factual/deterministic (no creativity)
    num_ctx=6000,       # Context window size (how much text it holds in memory)
    seed=24,            # Fixed seed for reproducible results
    num_predict=1024,   # Max tokens to generate in response
    num_thread=12,      # CPU threads to use
    num_gpu=1,          # Offload layers to GPU if available
    repeat_penalty=1.1, # Discourage repeating the same phrases
    top_k=40,           # Sampling parameter for diversity
    top_p=0.9,          # Sampling parameter for probability mass
    mirostat=0,         # Entropy-based sampling (off here)
    timeout=120,        # Crash if no response after 120s
)

tokenizer = tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))


def trim_to_token_limit(text: str, max_tokens: int) -> str:
    tokens = tokenizer.encode(text)
    if len(tokens) <= max_tokens:
        return text
    return tokenizer.decode(tokens[:max_tokens])


##  ##                                                                                            ##  ##  --  --  Tree Data Structure  --  --  ##  ##
@dataclass
class TreeNode:
    title: str
    node_id: str
    content: str
    summary: str = ""
    source_file: str = ""
    nodes: List["TreeNode"] = field(default_factory=list)

    def to_dict(self, include_content: bool = True) -> dict:
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "summary": self.summary,
            "source_file": self.source_file,
        }
        if include_content:
            result["content"] = self.content
        if self.nodes:
            result["nodes"] = [n.to_dict(include_content) for n in self.nodes]
        return result

    def node_count(self) -> int:
        return 1 + sum(c.node_count() for c in self.nodes)


class TreeSearchResult(BaseModel):
    thinking: str = Field(description="Reasoning about which nodes are relevant")
    node_list: List[str] = Field(description="List of relevant node IDs")


search_model = model.with_structured_output(TreeSearchResult)

##  ##                                                         ##  ##  --  --  Prompts  --  --  ##  ##
SUMMARY_PROMPT = """Extract core technical specifications from this section. 
Identify: Networking protocols such as OSI layers, Security controls, CLI syntax (PowerShell/Bash/Cisco iOS), NIST,
Hardware IDs, SQL schemas, or UX heuristics.
Maintain technical nomenclature, process/troubleshooting order and exact command flags.
Section: {title}

{content}"""

TREE_SEARCH_PROMPT = """You are a Senior Systems Architect. Analyze the document tree to locate 
specific technical context for the query. 
Route based on domain: Networking, OS, Security, or Dev.
Select nodes representing the highest granular technical detail.
Question: {query}

Document tree structure:
{tree_index}"""

ANSWER_PROMPT = """Act as a Tier 3 Engineer. Answer using the context provided following IT SOP standards.
1. Lead with CLI commands or configuration snippets if available.
2. Specify OS/Hardware dependencies and relevant standards.
3. If the context is insufficient, state:
'It appears there was some difficulty obtaining the content you requested.'
Question: {query}

Context:
{context}

Answer:"""


##  ##                                                ##  ##  --  --  File-to-Markdown Conversion  --  --  ##  ##
def convert_file_to_markdown(file_path: Path) -> str | None:
    """Route a file to the appropriate converter based on extension."""
    ext = file_path.suffix.lower()
    if ext == ".pdf":
        return _convert_pdf(file_path)
    elif ext == ".txt":
        return _convert_txt(file_path)
    else:
        console.print(f"[yellow]⚠ Unsupported file type: {file_path.name} — skipping[/yellow]")
        return None


def _convert_pdf(pdf_path: Path) -> str | None:
    """Convert a PDF to Markdown using pymupdf4llm (+ pymupdf_layout if installed)."""
    try:
        md_text = pymupdf4llm.to_markdown(str(pdf_path))
    except Exception as e:
        console.print(f"[red]✗ Failed to convert {pdf_path.name}: {e}[/red]")
        return None
    return _validate_and_chunk(md_text, pdf_path)


def _convert_txt(txt_path: Path) -> str | None:
    """Read a plaintext file and wrap it in markdown structure."""
    try:
        raw = txt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        console.print(f"[red]✗ Failed to read {txt_path.name}: {e}[/red]")
        return None
    return _validate_and_chunk(raw, txt_path)


def _validate_and_chunk(md_text: str, source_path: Path) -> str | None:
    """Check content quality and apply fallback chunking if headers are sparse."""
    if not md_text or len(md_text.strip()) < 50:
        console.print(f"[yellow]⚠ {source_path.name} produced empty/trivial content — skipping[/yellow]")
        return None

    header_count = len(re.findall(r"^#{1,3}\s+\S", md_text, re.MULTILINE))

    if header_count < MIN_HEADER_COUNT:
        console.print(
            f"[yellow]⚠ {source_path.name}: only {header_count} header(s) detected — "
            f"applying page-boundary fallback chunker[/yellow]"
        )
        md_text = _apply_fallback_chunking(md_text, source_path.stem)

    return md_text


def _apply_fallback_chunking(md_text: str, doc_name: str) -> str:
    """Insert ## Page N headers when the PDF produced no/few markdown headers.

    Splits on pymupdf4llm page-break markers (-----) or fixed token windows.
    """
    # pymupdf4llm inserts horizontal rules between pages
    pages = re.split(r"\n-{3,}\n", md_text)
    if len(pages) <= 1:
        # No page-break markers — fall back to token-window chunking
        pages = _chunk_by_tokens(md_text, FALLBACK_CHUNK_TOKENS)

    chunked_parts = [f"# {doc_name}\n"]
    for i, page in enumerate(pages, 1):
        page = page.strip()
        if not page:
            continue
        chunked_parts.append(f"## Page {i}\n\n{page}\n")

    return "\n".join(chunked_parts)


def _chunk_by_tokens(text: str, max_tokens: int) -> List[str]:
    """Split text into chunks of roughly max_tokens each, on paragraph boundaries."""
    paragraphs = text.split("\n\n")
    chunks, current_chunk, current_tokens = [], [], 0

    for para in paragraphs:
        para_tokens = count_tokens(para)
        if current_tokens + para_tokens > max_tokens and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk, current_tokens = [], 0
        current_chunk.append(para)
        current_tokens += para_tokens

    if current_chunk:
        chunks.append("\n\n".join(current_chunk))
    return chunks


##  ##                                                           ##  ##  --  --  Tree Builder  --  --  ##  ##
def _has_meaningful_content(text: str, min_length: int = 40) -> bool:
    if not text:
        return False
    stripped = re.sub(r"^#+\s+.*$", "", text, flags=re.MULTILINE)
    stripped = stripped.strip()
    return len(stripped) >= min_length


def build_tree(markdown: str, doc_title: str = "Document", start_id: int = 1) -> Tuple[TreeNode, int]:
    """Build a hierarchical tree from markdown. Returns (root, next_available_id)."""
    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "title"), ("##", "section"), ("###", "subsection")],
        strip_headers=False,
    )
    sections = splitter.split_text(markdown)
    root = TreeNode(title=doc_title, node_id=f"{start_id - 1:04d}", content="", source_file=doc_title)
    counter = start_id
    stack: List[Tuple[int, TreeNode]] = [(0, root)]
    levels = {"title": 1, "section": 2, "subsection": 3}

    for section in sections:
        level, title = 0, "General"
        for key, val in levels.items():
            if key in section.metadata:
                level, title = val, section.metadata[key]
        if level == 0:
            root.content += section.page_content
            continue
        node = TreeNode(
            title=title,
            node_id=f"{counter:04d}",
            content=section.page_content,
            source_file=doc_title,
        )
        counter += 1
        while len(stack) > 1 and stack[-1][0] >= level:
            stack.pop()
        stack[-1][1].nodes.append(node)
        stack.append((level, node))

    return root, counter


def summarize_tree(node: TreeNode):
    """Generate LLM summaries bottom-up: leaf nodes first, then parents."""
    for child in node.nodes:
        summarize_tree(child)
    has_content = _has_meaningful_content(node.content)
    has_child_summaries = any(c.summary for c in node.nodes)
    if not has_content and not has_child_summaries:
        return
    if has_child_summaries:
        children_text = "\n".join(
            f"- {c.title}: {c.summary}" for c in node.nodes if c.summary
        )
        text = (
            f"{node.content}\n\nChild Sections:\n{children_text}"
            if has_content else children_text
        )
    else:
        text = node.content
    trimmed = trim_to_token_limit(text, MAX_SUMMARY_TOKENS)
    try:
        node.summary = model.invoke(
            SUMMARY_PROMPT.format(title=node.title, content=trimmed)
        ).content.strip()
    except Exception as e:
        console.print(f"[red]  ✗ Summarize failed for '{node.title}': {e}[/red]")
        node.summary = f"[Summary unavailable: {node.title}]"


##  ##                                                             ##  ##  --  --  Node Map & Context  --  --  ##  ##
def create_node_map(node: TreeNode) -> Dict[str, TreeNode]:
    m = {node.node_id: node}
    for c in node.nodes:
        m.update(create_node_map(c))
    return m


def build_context(node_map: Dict[str, TreeNode], node_ids: List[str]) -> str:
    total_tokens = 0
    context_parts = []
    for nid in node_ids[:MAX_RETRIEVAL_NODES]:
        if nid not in node_map:
            continue
        content = node_map[nid].content
        tokens = count_tokens(content)
        if total_tokens + tokens > MAX_CONTEXT_TOKENS:
            remaining = MAX_CONTEXT_TOKENS - total_tokens
            if remaining <= 0:
                break
            content = trim_to_token_limit(content, remaining)
            context_parts.append(content)
            break
        context_parts.append(content)
        total_tokens += tokens
    return "\n\n".join(context_parts)


##  ##                                                           ##  ##  --  --  Retrieval & Answer  --  --  ##  ##
def retrieve_and_answer(tree: TreeNode, query: str):
    process = psutil.Process(os.getpid())
    mem_mb = process.memory_info().rss / (1024 * 1024)
    if mem_mb > 10000:
        raise MemoryError("Process exceeded 10GB RAM safety threshold")

    node_map = create_node_map(tree)
    tree_index = json.dumps(tree.to_dict(False))
    search_res = search_model.invoke(
        TREE_SEARCH_PROMPT.format(query=query, tree_index=tree_index)
    )
    context = build_context(node_map, search_res.node_list)
    answer = model.invoke(
        ANSWER_PROMPT.format(query=query, context=context)
    ).content.strip()
    return answer, search_res.node_list, search_res.thinking


##  ##                                                           ##  ##  --  --  Serialization  --  --  ##  ##
def save_tree(tree: TreeNode, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(tree.to_dict(), f, indent=2, ensure_ascii=False)
    console.print(f"[green]✔ Saved tree → {path.name}[/green]")


def load_tree(path: Path) -> TreeNode:
    with open(path) as f:
        return _dict_to_node(json.load(f))


def _dict_to_node(d: dict) -> TreeNode:
    n = TreeNode(
        title=d["title"],
        node_id=d["node_id"],
        content=d.get("content", ""),
        summary=d.get("summary", ""),
        source_file=d.get("source_file", ""),
    )
    n.nodes = [_dict_to_node(c) for c in d.get("nodes", [])]
    return n


##  ##                                                  ##  ##  --  --  Manifest (tracks processed PDFs)  --  --  ##  ##
def load_manifest() -> Dict[str, float]:
    """Returns {filename: mtime} for already-processed PDFs."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE) as f:
            return json.load(f)
    return {}


def save_manifest(manifest: Dict[str, float]):
    with open(MANIFEST_FILE, "w") as f:
        json.dump(manifest, f, indent=2)


##  ##                                                      ##  ##  --  --  Library Scanner & Ingestion Pipeline  --  --  ##  ##
def scan_and_ingest_library() -> TreeNode | None:
    """Scan Library/ for PDFs and TXT files, convert → build tree → summarize. Returns merged root."""
    supported_files = sorted(
        list(LIBRARY_DIR.glob("*.pdf")) + list(LIBRARY_DIR.glob("*.txt"))
    )
    if not supported_files:
        console.print("[yellow]⚠ No supported files (PDF/TXT) found in Library/ — nothing to ingest.[/yellow]")
        return None

    manifest = load_manifest()
    new_files = []
    for fpath in supported_files:
        mtime = fpath.stat().st_mtime
        if fpath.name in manifest and manifest[fpath.name] == mtime:
            console.print(f"[dim]  ↳ Already processed: {fpath.name}[/dim]")
            continue
        new_files.append((fpath, mtime))

    if not new_files:
        console.print("[green]✔ All Library PDFs already ingested — no new files.[/green]")
        return None

    console.print(f"\n[bold cyan]Found {len(new_files)} new/modified file(s) to ingest[/bold cyan]\n")

    # If we have an existing tree, load it so we can merge
    if KB_FILE.exists():
        master_root = load_tree(KB_FILE)
        next_id = _max_node_id(master_root) + 1
    else:
        master_root = TreeNode(title="Holonet Root", node_id="0000", content="")
        next_id = 1

    for file_path, mtime in new_files:
        console.print(Panel(f"[bold]{file_path.name}[/bold]", border_style="cyan"))
        t0 = time.time()

        # Step 1: Convert to Markdown
        console.print(f"  [cyan]Step 1:[/cyan] Converting {file_path.suffix.upper()} → Markdown…")
        md_text = convert_file_to_markdown(file_path)
        if md_text is None:
            continue

        # Cache the markdown
        md_cache_path = MD_CACHE_DIR / f"{file_path.stem}.md"
        md_cache_path.write_text(md_text, encoding="utf-8")
        console.print(f"  [dim]  ↳ Cached markdown → {md_cache_path.name} ({len(md_text):,} chars)[/dim]")

        # Step 2: Build tree
        console.print("  [cyan]Step 2:[/cyan] Building tree index…")
        doc_tree, next_id = build_tree(md_text, doc_title=file_path.stem, start_id=next_id)
        node_count = doc_tree.node_count()
        console.print(f"  [dim]  ↳ Tree built: {node_count} nodes[/dim]")

        # Step 3: Summarize
        console.print("  [cyan]Step 3:[/cyan] Generating node summaries (this takes a while)…")
        summarize_tree(doc_tree)
        elapsed = time.time() - t0
        console.print(f"  [green]✔ {file_path.name} ingested in {elapsed:.1f}s[/green]\n")

        # Merge into master root
        master_root.nodes.append(doc_tree)

        # Update manifest immediately (so partial runs don't redo completed files)
        manifest[file_path.name] = mtime
        save_manifest(manifest)

    # Save the merged tree
    save_tree(master_root, KB_FILE)
    return master_root


def _max_node_id(node: TreeNode) -> int:
    """Find the highest numeric node_id in the tree."""
    try:
        max_id = int(node.node_id)
    except ValueError:
        max_id = 0
    for c in node.nodes:
        max_id = max(max_id, _max_node_id(c))
    return max_id


##  ##                                                               ##  ##  --  --  Display  --  --  ##  ##
def display_tree(node: TreeNode, parent: RichTree | None = None, depth: int = 0) -> RichTree:
    """Render tree using rich, with depth limiting for readability."""
    label = f"[bold]{node.title}[/bold] [dim]({node.node_id})[/dim]"
    if node.summary:
        short = node.summary[:120] + ("…" if len(node.summary) > 120 else "")
        label += f"\n[italic]{short}[/italic]"

    branch = parent.add(label) if parent else RichTree(label)

    if depth < 4:  # prevent terminal flood on deep trees
        for child in node.nodes:
            display_tree(child, branch, depth + 1)

    return branch


# ══════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    console.print(Panel(
        "[bold cyan]HOLONET VECTORLESS RAG[/bold cyan]\n"
        "[dim]Tree-indexed retrieval • gemma3:1b • No embeddings[/dim]",
        border_style="bright_cyan",
    ))

    ##  ##                                                 ##  ##  --  --  Phase 1: Load or Build  --  --  ##  ##
    tree = None

    if KB_FILE.exists():
        console.print("[bold cyan]Loading existing knowledge tree…[/bold cyan]")
        tree = load_tree(KB_FILE)
        console.print(f"[green]✔ Loaded tree: {tree.node_count()} nodes[/green]\n")

    # Always scan for new/modified PDFs regardless of existing tree
    console.print("[bold cyan]Scanning Library/ for new PDFs…[/bold cyan]")
    updated_tree = scan_and_ingest_library()
    if updated_tree is not None:
        tree = updated_tree

    if tree is None:
        console.print("[red]✗ No knowledge tree available. Add PDFs to Knowledgebase_Alpha/Library/ and re-run.[/red]")
        exit(1)

    ##  ##                                               ##  ##  --  --  Phase 2: Display Tree Summary  --  --  ##  ##
    node_map = create_node_map(tree)

    sample = {
        nid: {
            "title": node.title,
            "summary": (node.summary[:100] + "…") if len(node.summary) > 100 else node.summary,
            "content_length": len(node.content),
            "children": len(node.nodes),
        }
        for nid, node in list(node_map.items())[:MAX_NODES_DISPLAY]
    }

    console.print(Panel("[bold]Node Map Sample[/bold]", border_style="blue"))
    console.print(Syntax(json.dumps(sample, indent=2), "json", theme="one-dark"))
    console.print()
    console.print(display_tree(tree))
    console.print("\n" + "─" * 80 + "\n")

    ##  ##                                                  ##  ##  --  --  Phase 3: Query Loop  --  --  ##  ##
    queries = [
        "Describe 5 built-in exceptions to use in Python code",
        "Describe Red Teaming Solution Framework for Generative AI",
        "List the import modules used for automating cybersecurity tasks",
        "Unknown query test: What is the weather in Tatooine?",  # grounding check
    ]

    console.print(Panel(
        f"[bold cyan]Running {len(queries)} test queries[/bold cyan]",
        border_style="bright_cyan",
    ))

    for i, query in enumerate(queries, 1):
        console.print(f"\n[bold]Query {i}/{len(queries)}:[/bold] {query}\n")
        try:
            answer, node_ids, thinking = retrieve_and_answer(tree, query)
            console.print(Panel(thinking, title="Architect Reasoning", border_style="yellow"))
            console.print(f"[bold]Retrieved Nodes:[/bold]")
            for nid in node_ids:
                if nid in node_map:
                    console.print(f"  [cyan]{nid}[/cyan] — {node_map[nid].title}")
            console.print(f"\n[bold green]Answer:[/bold green] {answer}\n")
        except Exception as e:
            console.print(f"[red]✗ Query failed: {e}[/red]\n")
        console.print("─" * 80)
