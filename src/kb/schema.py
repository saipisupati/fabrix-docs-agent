"""
kb/schema.py: structured knowledge-base record types for Fabrix public docs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Topic:
    id: str
    name: str
    section: str
    summary: str
    doc_paths: list[str] = field(default_factory=list)
    related_topics: list[str] = field(default_factory=list)


@dataclass
class Entity:
    id: str
    kind: str  # bot | extension | pipeline | guide | integration
    name: str
    summary: str
    section: str
    source: str
    url: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Fact:
    id: str
    text: str
    source: str
    url: str
    section: str
    entity_id: str | None = None
    example: str | None = None


@dataclass
class Procedure:
    id: str
    title: str
    steps: list[str]
    source: str
    url: str
    section: str


@dataclass
class Relation:
    id: str
    from_id: str
    to_id: str
    relation: str  # e.g. bot_in_extension, related_topic


@dataclass
class KnowledgeBase:
    topics: list[Topic] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    procedures: list[Procedure] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    version: str = "1"
    embedding_model: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "embedding_model": self.embedding_model,
            "topics": [asdict(t) for t in self.topics],
            "entities": [asdict(e) for e in self.entities],
            "facts": [asdict(f) for f in self.facts],
            "procedures": [asdict(p) for p in self.procedures],
            "relations": [asdict(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "KnowledgeBase":
        return cls(
            version=data.get("version", "1"),
            embedding_model=data.get("embedding_model", ""),
            topics=[Topic(**t) for t in data.get("topics", [])],
            entities=[Entity(**e) for e in data.get("entities", [])],
            facts=[Fact(**f) for f in data.get("facts", [])],
            procedures=[Procedure(**p) for p in data.get("procedures", [])],
            relations=[Relation(**r) for r in data.get("relations", [])],
        )

    def searchable_entries(self) -> list[dict[str, Any]]:
        """Flat list of embeddable KB cards used at retrieval time."""
        entries: list[dict[str, Any]] = []
        for t in self.topics:
            entries.append({
                "id": f"topic:{t.id}",
                "kind": "topic",
                "title": t.name,
                "text": f"{t.name}. {t.summary}",
                "source": t.doc_paths[0] if t.doc_paths else "",
                "url": "",
                "section": t.section,
                "example": "",
            })
        for e in self.entities:
            meta = e.metadata or {}
            text = f"{e.kind} {e.name}. {e.summary}"
            # Phase 3: include structured bot params so retrieve_kb surfaces param tables
            params = meta.get("parameters") or []
            if e.kind == "bot" and params:
                names = meta.get("param_names") or [p.get("name") for p in params if p.get("name")]
                bits = []
                for p in params[:20]:
                    n = p.get("name") or ""
                    if not n:
                        continue
                    req = "required" if p.get("required") else "optional"
                    bits.append(f"{n} ({req})")
                text = (
                    f"bot {e.name} parameters {', '.join(names[:16])}. "
                    f"{e.summary}. Details: {'; '.join(bits)}"
                )[:4000]
            entries.append({
                "id": f"entity:{e.id}",
                "kind": e.kind,
                "title": e.name,
                "text": text,
                "source": e.source,
                "url": e.url,
                "section": e.section,
                "example": meta.get("example", "") or "",
            })
        for f in self.facts:
            entries.append({
                "id": f"fact:{f.id}",
                "kind": "fact",
                "title": f.text[:80],
                "text": f.text + (f" Example: {f.example}" if f.example else ""),
                "source": f.source,
                "url": f.url,
                "section": f.section,
                "example": f.example or "",
            })
        for p in self.procedures:
            step_text = " ".join(f"{i+1}. {s}" for i, s in enumerate(p.steps[:12]))
            entries.append({
                "id": f"procedure:{p.id}",
                "kind": "procedure",
                "title": p.title,
                "text": f"{p.title}. Steps: {step_text}",
                "source": p.source,
                "url": p.url,
                "section": p.section,
                "example": step_text[:400],
            })
        return entries
