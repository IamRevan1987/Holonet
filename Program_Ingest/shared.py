# Shared Data Structures and Constants for Holonet
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
BASE_DIR = Path.home() / "docker/APIinterface/Holonet"
KB_DIR = BASE_DIR / "Knowledgebase_Alpha"
LIBRARY_DIR = KB_DIR / "Library"
MD_CACHE_DIR = KB_DIR / "markdown_cache"
KB_FILE = KB_DIR / "holonet_resources.json"
MANIFEST_FILE = KB_DIR / "processed_files.json"

# Ensure directories exist
KB_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Data Structures
# ─────────────────────────────────────────────
@dataclass
class TreeNode:
    """
    Represents a single section or header in the document.
    Think of this like a folder in a file system, but for text.
    """
    title: str                  # The header text (e.g., "1. Introduction")
    node_id: str                # Unique ID (e.g., "0001") for tracking
    content: str                # The text content directly under this header
    summary: str = ""           # LLM-generated summary of this section + children
    source_file: str = ""       # Name of the PDF this came from
    domain: str = ""            # Detected domain label (Networking, Security, OS, Dev, General)
    nodes: List["TreeNode"] = field(default_factory=list)  # Child subsections

    def to_dict(self, include_content: bool = True) -> dict:
        """Convert the tree to a dictionary (JSON-ready format)."""
        result = {
            "title": self.title,
            "node_id": self.node_id,
            "summary": self.summary,
            "source_file": self.source_file,
            "domain": self.domain,
        }
        if include_content:
            result["content"] = self.content
        if self.nodes:
            result["nodes"] = [n.to_dict(include_content) for n in self.nodes]
        return result

    def node_count(self) -> int:
        """Recursively count how many nodes (sections) are in this branch."""
        return 1 + sum(c.node_count() for c in self.nodes)

def _dict_to_node(d: dict) -> TreeNode:
    n = TreeNode(
        title=d["title"],
        node_id=d["node_id"],
        content=d.get("content", ""),
        summary=d.get("summary", ""),
        source_file=d.get("source_file", ""),
        domain=d.get("domain", ""),
    )
    n.nodes = [_dict_to_node(c) for c in d.get("nodes", [])]
    return n

def load_tree(path: Path) -> TreeNode:
    with open(path) as f:
        return _dict_to_node(json.load(f))

def create_node_map(node: TreeNode) -> Dict[str, TreeNode]:
    m = {node.node_id: node}
    for c in node.nodes:
        m.update(create_node_map(c))
    return m
