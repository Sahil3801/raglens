from dataclasses import dataclass, field
from typing import Dict, Any, List

@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RetrievalResult:
    query: str
    chunks: List[DocumentChunk]