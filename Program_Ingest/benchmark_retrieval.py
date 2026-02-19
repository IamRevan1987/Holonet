"""
Benchmark Retrieval & Domain Quality
====================================
Analyzes the current Holonet Knowledge Graph to measure:
1. Domain distribution (Networking vs Security vs OS vs Dev)
2. Granularity (Avg tokens per node)
3. Retrieval Alignment (Do Networking queries fetch Networking nodes?)

Usage:
  python Program_Ingest/benchmark_retrieval.py
"""

import json
import time
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from ingest import load_tree, create_node_map, retrieve_and_answer, TreeNode, count_tokens, KB_FILE

console = Console()

# ─────────────────────────────────────────────────────────────
#  Test Queries (Mapped to Expected Domains)
# ─────────────────────────────────────────────────────────────
BENCHMARK_QUERIES = [
    ("How does the OSI model layer 3 handle routing?", "Networking"),
    ("What are the core principles of NIST SP 800-53?", "Security"),
    ("Explain Python exception handling with try/except blocks.", "Dev"),
    ("How do I change file permissions in Linux using chmod?", "OS"),
    ("What is the difference between TCP and UDP?", "Networking"),
    ("List common OWASP Top 10 vulnerabilities.", "Security"),
]

def analyze_tree_stats(tree: TreeNode):
    """Print structural stats about the knowledge graph."""
    node_map = create_node_map(tree)
    total_nodes = len(node_map)
    
    domain_counts = {"Networking": 0, "Security": 0, "OS": 0, "Dev": 0, "General": 0, "Unknown": 0}
    domain_tokens = {"Networking": 0, "Security": 0, "OS": 0, "Dev": 0, "General": 0, "Unknown": 0}

    for node in node_map.values():
        d = node.domain if node.domain else "Unknown"
        if d not in domain_counts: 
            d = "Unknown" # Safety bucket
        
        domain_counts[d] += 1
        domain_tokens[d] += count_tokens(node.content)

    # ─── Table 1: Domain Distribution ───
    table = Table(title="Knowledge Graph Domain Distribution", border_style="cyan")
    table.add_column("Domain", style="cyan", justify="left")
    table.add_column("Nodes", style="magenta", justify="right")
    table.add_column("% of Graph", style="green", justify="right")
    table.add_column("Avg Tokens/Node", style="yellow", justify="right")

    for domain, count in domain_counts.items():
        if count == 0: continue
        pct = (count / total_nodes) * 100
        avg_tok = domain_tokens[domain] / count
        table.add_row(domain, str(count), f"{pct:.1f}%", f"{avg_tok:.0f}")

    console.print(table)
    console.print(f"[dim]Total Nodes: {total_nodes}[/dim]\n")

def run_retrieval_benchmark(tree: TreeNode):
    """Run predefined queries and check if retrieved nodes match the expected domain."""
    
    table = Table(title="Retrieval Domain Alignment", border_style="blue")
    table.add_column("Query", style="white", justify="left", overflow="fold")
    table.add_column("Target", style="cyan", justify="center")
    table.add_column("Retrieved Domains", style="magenta", justify="left")
    table.add_column("Alignment Score", style="green", justify="right")

    node_map = create_node_map(tree)

    for query, target_domain in BENCHMARK_QUERIES:
        # We only care about retrieval, so we can ignore the LLM answer generation time mostly,
        # but retrieve_and_answer does both. We'll measure total time.
        t0 = time.time()
        try:
            _, node_ids, _ = retrieve_and_answer(tree, query)
        except Exception as e:
            console.print(f"[red]Error running query '{query}': {e}[/red]")
            continue
            
        retrieved_domains = []
        matches = 0
        for nid in node_ids:
            if nid in node_map:
                d = node_map[nid].domain
                retrieved_domains.append(d if d else "General")
                if d == target_domain:
                    matches += 1
        
        score = (matches / len(node_ids)) * 100 if node_ids else 0
        score_str = f"{score:.0f}%"
        
        # Color code the score
        if score == 100: score_str = f"[green]{score_str}[/green]"
        elif score >= 50: score_str = f"[yellow]{score_str}[/yellow]"
        else: score_str = f"[red]{score_str}[/red]"

        table.add_row(query, target_domain, ", ".join(retrieved_domains), score_str)

    console.print(table)


if __name__ == "__main__":
    console.print(Panel("[bold]Holonet Ingestion Benchmark[/bold]", border_style="cyan"))
    
    if not KB_FILE.exists():
        console.print("[red]✗ Knowledge base not found. Run ingest.py first.[/red]")
        exit(1)

    console.print("Loading Knowledge Graph...")
    tree = load_tree(KB_FILE)
    
    analyze_tree_stats(tree)
    run_retrieval_benchmark(tree)
