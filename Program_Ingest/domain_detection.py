"""
Domain Shift Detection Layer
=============================
Detects topic shifts inside a single header section and splits accordingly.
Uses regex-based pattern matching — no LLM calls needed.

Each paragraph is scored against domain-specific signal patterns.
Adjacent same-domain paragraphs are merged into contiguous DomainBlocks.

Supported domains:  Networking | Security | OS | Dev | General (fallback)
"""

import re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

# ─────────────────────────────────────────────────────────────
#  Domain Signal Patterns
# ─────────────────────────────────────────────────────────────
# Each key is a domain name; each value is a list of regex patterns.
# A paragraph is scored by counting how many patterns match.

DOMAIN_PATTERNS: Dict[str, List[str]] = {
    "Networking": [
        r"\bOSI\b",
        r"\bTCP/IP\b",
        r"\bTCP\b",
        r"\bUDP\b",
        r"\bDNS\b",
        r"\bDHCP\b",
        r"\bBGP\b",
        r"\bOSPF\b",
        r"\bEIGRP\b",
        r"\bVLAN\b",
        r"\bsubnet\w*\b",
        r"\bLayer\s*[1-7]\b",
        r"\bIPv[46]\b",
        r"\bIP\s+address\b",
        r"\bpacket\b",
        r"\bfirewall\b",
        r"\bswitch(?:es|ing)?\b",
        r"\brouter\b",
        r"\brouting\b",
        r"\bping\b",
        r"\btraceroute\b",
        r"\bnslookup\b",
        r"\bARP\b",
        r"\bNAT\b",
        r"\bHTTP[S]?\b",
        r"\bSSH\b",
        r"\bTelnet\b",
        r"\bSNMP\b",
        r"\bport\s+\d+\b",
        r"\bgateway\b",
        r"\bEthernet\b",
        r"\bWi-?Fi\b",
        r"\bbandwidth\b",
        r"\blatency\b",
        r"\bthroughput\b",
        r"\bsocket\b",
    ],
    "Security": [
        r"\bNIST\b",
        r"\bCVE-\d+",
        r"\bCIS\b",
        r"\bMITRE\b",
        r"\bATT&CK\b",
        r"\bencrypt\w*\b",
        r"\bdecrypt\w*\b",
        r"\bIDS\b",
        r"\bIPS\b",
        r"\bSIEM\b",
        r"\bpenetration\s+test\w*\b",
        r"\bred\s+team\w*\b",
        r"\bblue\s+team\w*\b",
        r"\bvulnerabilit\w+\b",
        r"\bSP\s*800-\d+",
        r"\bZero\s*Trust\b",
        r"\bACL\b",
        r"\bthreat\s+(actor|model|hunt)\w*\b",
        r"\bmalware\b",
        r"\bransomware\b",
        r"\bphishing\b",
        r"\bsocial\s+engineering\b",
        r"\bauthenticat\w+\b",
        r"\bauthoriz\w+\b",
        r"\bforensic\w*\b",
        r"\bincident\s+response\b",
        r"\brisk\s+(assessment|management|analysis)\b",
        r"\bcompliance\b",
        r"\bOWASP\b",
        r"\bsecurity\s+control\w*\b",
        r"\bISO\s*27\d{3}\b",
        r"\bPCI[\s-]DSS\b",
        r"\bharden\w*\b",
        r"\bexploit\w*\b",
    ],
    "OS": [
        r"\bbash\b",
        r"\bBourne\b",
        r"\bPowerShell\b",
        r"\bcmd\.exe\b",
        r"\bterminal\b",
        r"\bshell\b",
        r"\bsudo\b",
        r"\bchmod\b",
        r"\bchown\b",
        r"\bgrep\b",
        r"\bawk\b",
        r"\bsed\b",
        r"\bsystemctl\b",
        r"\bregistry\b",
        r"\bkernel\b",
        r"\bdaemon\b",
        r"\bcrontab\b",
        r"\bcron\b",
        r"\bGet-\w+\b",
        r"\bSet-\w+\b",
        r"\bNew-\w+\b",
        r"\bRemove-\w+\b",
        r"\bfile\s*system\b",
        r"\bext[234]\b",
        r"\bNTFS\b",
        r"\bLinux\b",
        r"\bWindows\b",
        r"\bmacOS\b",
        r"\bprocess(?:es)?\b",
        r"\bthread\b",
        r"\bmount\b",
        r"\bpartition\b",
        r"\bboot\s*loader\b",
        r"\bGRUB\b",
        r"\bsystemd\b",
        r"\binit\.d\b",
        r"\bls\s+-\w+",
        r"\bcd\s+/",
        r"\bmkdir\b",
        r"\brm\s+-",
        r"\bcp\s+-",
        r"\bcat\s+",
    ],
    "Dev": [
        r"\bdef\s+\w+",
        r"\bclass\s+\w+",
        r"\bimport\s+\w+",
        r"\bfrom\s+\w+\s+import\b",
        r"\btry\s*:",
        r"\bexcept\s+\w+",
        r"\braise\s+\w+",
        r"\bSQL\b",
        r"\bAPI\b",
        r"\bJSON\b",
        r"\bREST\w*\b",
        r"\bfunction\b",
        r"\breturn\b",
        r"\bgit\s+(commit|push|pull|clone|merge|rebase)\b",
        r"\bpip\s+install\b",
        r"\bnpm\s+install\b",
        r"\bfor\s+\w+\s+in\b",
        r"\bwhile\s+\w+",
        r"\bif\s+__name__\b",
        r"\bprint\s*\(",
        r"\bvariable\b",
        r"\bstring\b",
        r"\binteger\b",
        r"\bfloat\b",
        r"\blist\b",
        r"\bdictionary\b",
        r"\btuple\b",
        r"\bboolean\b",
        r"\bloop\b",
        r"\brecursion\b",
        r"\binheritance\b",
        r"\bpolymorphism\b",
        r"\bencapsulation\b",
        r"\bexception\b",
        r"\bdebugg\w*\b",
        r"\bunit\s+test\w*\b",
        r"\bcompil\w+\b",
        r"\binterpret\w+\b",
    ],
}

# ─────────────────────────────────────────────────────────────
#  CLI Syntax Detectors (specialised OS subcategory)
# ─────────────────────────────────────────────────────────────
# These boost the OS score when specific CLI syntax is detected.

CLI_PATTERNS: Dict[str, List[str]] = {
    "PowerShell": [
        r"\bGet-\w+\b",
        r"\bSet-\w+\b",
        r"\bNew-\w+\b",
        r"\bRemove-\w+\b",
        r"\bInvoke-\w+\b",
        r"\b\$\w+\b",             # PowerShell variables
        r"\b-\w+Parameter\b",
        r"\bWrite-Host\b",
    ],
    "Bash": [
        r"\bsudo\s+\w+",
        r"\bchmod\s+\d{3}\b",
        r"\bgrep\s+-\w+",
        r"\bawk\s+'",
        r"\bsed\s+-",
        r"\bpipe\b|\|",
        r"\b\./\w+\.sh\b",
        r"#!/bin/(ba)?sh",
    ],
    "Cisco_IOS": [
        r"\benable\b",
        r"\bconfigure\s+terminal\b",
        r"\bshow\s+(ip|running|version|interface)\b",
        r"\binterface\s+(Gig|Fast|Ethernet|Loopback)\w*",
        r"\brouter\s+(ospf|eigrp|bgp)\b",
        r"\bno\s+shutdown\b",
        r"\bip\s+route\b",
        r"\bhostname\b",
    ],
}

# ─────────────────────────────────────────────────────────────
#  Standards / RFC / NIST Detectors
# ─────────────────────────────────────────────────────────────
# These boost Security or Networking scores depending on the standard.

STANDARDS_PATTERNS: Dict[str, Tuple[str, List[str]]] = {
    # standard_name -> (target_domain, patterns)
    "NIST": ("Security", [
        r"\bNIST\s+SP\s*\d+",
        r"\bSP\s*800-\d+",
        r"\bNIST\s+(CSF|RMF|Cybersecurity\s+Framework)\b",
        r"\bFIPS\s+\d+",
    ]),
    "RFC": ("Networking", [
        r"\bRFC\s*\d{3,5}\b",
    ]),
    "ISO": ("Security", [
        r"\bISO\s*(\/IEC\s*)?27\d{3}\b",
        r"\bISO\s*31000\b",
    ]),
    "CIS": ("Security", [
        r"\bCIS\s+(Benchmark|Control|Critical)\w*\b",
    ]),
    "OWASP": ("Security", [
        r"\bOWASP\s+(Top\s*10|ASVS|SAMM)\b",
    ]),
}


# ─────────────────────────────────────────────────────────────
#  DomainBlock (structured result)
# ─────────────────────────────────────────────────────────────

@dataclass
class DomainBlock:
    """A contiguous block of text assigned to a single domain."""
    domain: str               # "Networking", "Security", "OS", "Dev", or "General"
    text: str                 # The paragraph(s) belonging to this block
    confidence: float = 0.0   # Highest match score (higher = more signals matched)


# ─────────────────────────────────────────────────────────────
#  Internal: Classify a single paragraph
# ─────────────────────────────────────────────────────────────

def _classify_paragraph(paragraph: str) -> Tuple[str, float]:
    """
    Score a paragraph against all domain pattern dictionaries.
    Returns (domain_name, score).  Falls back to "General" if no patterns match.
    """
    if not paragraph or not paragraph.strip():
        return ("General", 0.0)

    scores: Dict[str, float] = {domain: 0.0 for domain in DOMAIN_PATTERNS}

    # 1. Score against main domain patterns
    for domain, patterns in DOMAIN_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, paragraph, re.IGNORECASE)
            scores[domain] += len(matches)

    # 2. Boost from CLI syntax detectors → adds to OS score
    for cli_type, patterns in CLI_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, paragraph, re.IGNORECASE)
            if matches:
                # Cisco IOS boosts both OS and Networking
                if cli_type == "Cisco_IOS":
                    scores["OS"] += len(matches) * 0.5
                    scores["Networking"] += len(matches) * 0.5
                else:
                    scores["OS"] += len(matches)

    # 3. Boost from standards detectors
    for std_name, (target_domain, patterns) in STANDARDS_PATTERNS.items():
        for pattern in patterns:
            matches = re.findall(pattern, paragraph, re.IGNORECASE)
            if matches:
                scores[target_domain] += len(matches) * 1.5  # Standards get extra weight

    # Find the top-scoring domain
    top_domain = max(scores, key=scores.get)
    top_score = scores[top_domain]

    if top_score == 0:
        return ("General", 0.0)

    return (top_domain, top_score)


# ─────────────────────────────────────────────────────────────
#  Internal: Merge adjacent same-domain blocks
# ─────────────────────────────────────────────────────────────

def _merge_adjacent(blocks: List[DomainBlock]) -> List[DomainBlock]:
    """
    Collapse consecutive blocks with the same domain label
    into a single block.  Confidence = max of merged blocks.
    """
    if not blocks:
        return []

    merged: List[DomainBlock] = [blocks[0]]

    for block in blocks[1:]:
        if block.domain == merged[-1].domain:
            # Same domain as previous — merge text
            merged[-1].text += "\n\n" + block.text
            merged[-1].confidence = max(merged[-1].confidence, block.confidence)
        else:
            merged.append(block)

    return merged


# ─────────────────────────────────────────────────────────────
#  Public API: detect_domain_shift()
# ─────────────────────────────────────────────────────────────

def detect_domain_shift(text: str) -> List[DomainBlock]:
    """
    Main entry point.  Scans paragraphs in the input text, assigns
    domain labels, merges adjacent same-domain blocks.

    Args:
        text: The text content of a single header section.

    Returns:
        Ordered list of DomainBlocks.  If the entire section
        is one domain, returns a single-element list.
    """
    if not text or not text.strip():
        return []

    # Split into paragraphs (double newline boundary)
    paragraphs = re.split(r"\n{2,}", text.strip())

    # Classify each paragraph
    raw_blocks: List[DomainBlock] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        domain, score = _classify_paragraph(para)
        raw_blocks.append(DomainBlock(domain=domain, text=para, confidence=score))

    # Merge adjacent same-domain blocks
    return _merge_adjacent(raw_blocks)
