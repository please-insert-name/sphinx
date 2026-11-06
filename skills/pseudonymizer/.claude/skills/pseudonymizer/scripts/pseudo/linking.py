"""Entitätsverknüpfung und Propagation.

* Gleiche Entität -> gleicher Platzhalter (stabile, typisierte Platzhalter erhalten
  Koreferenz für das nachgelagerte LLM; LLM-Redactor arXiv:2604.12064; ANOPPI/PSILENCE).
* Namensvarianten: "Dr. Harald Schneider" ~ "Herr Schneider" ~ "Harald Schneider";
  verschiedene Vornamen bei gleichem Nachnamen -> verschiedene Personen.
* Propagation (korpusweite Pseudonymisierung, vgl. Masketeer): einmal erkannte Werte
  werden an allen weiteren Stellen des Dokuments ebenfalls ersetzt.
"""
from __future__ import annotations

import re

from .common import Span, find_all

_LEGAL = re.compile(
    r"\b(?:gmbh\s?&\s?co\.?\s?kg|gmbh|ag|kg|ohg|gbr|ug(?:\s?\(haftungsbeschränkt\))?|e\.?\s?v\.?|mbh|se|kgaa|"
    r"corp\.?|corporation|inc\.?|ltd\.?|llc|plc|co\.?|limited)(?=\W|$)", re.I)


def _norm(typ: str, value: str) -> str:
    v = " ".join(value.split())
    if typ == "TELEFON":
        d = re.sub(r"\D", "", v)
        if v.startswith("+49"):
            d = "0" + d[2:]
        elif d.startswith("0049"):
            d = "0" + d[4:]
        return d
    if typ in ("IBAN", "KARTE", "ID", "KFZ"):
        return re.sub(r"[^0-9A-Za-z]", "", v).upper()
    if typ == "CREDENTIAL":
        return v
    if typ == "ORG":
        k = _LEGAL.sub("", v.casefold())
        k = re.sub(r"[^\w]", "", k)
        return k or v.casefold()
    return v.casefold()


def _first_compatible(a: list[str], b: list[str]) -> bool:
    x, y = a[0].casefold().rstrip("."), b[0].casefold().rstrip(".")
    if x == y:
        return True
    if len(x) == 1 or len(y) == 1:         # Initiale: "H. Schneider" ~ "Harald Schneider"
        return x[0] == y[0]
    return False


class Linker:
    """Vergibt Platzhalter-Schlüssel und hält die Zuordnungstabelle (Tresor-Inhalt)."""

    def __init__(self, data: dict | None = None):
        data = data or {}
        self.entities: dict[str, dict] = data.get("entities", {})
        self.counters: dict[str, int] = data.get("counters", {})
        self.tick = data.get("tick", 0)
        self.index: dict[tuple[str, str], str] = {}
        self.ambiguous = 0
        for key, ent in self.entities.items():
            ent["variants"] = list(dict.fromkeys(ent.get("variants", [])))
            for var in ent["variants"]:
                self.index[(ent["type"], _norm(ent["type"], var))] = key

    def to_dict(self) -> dict:
        return {"entities": self.entities, "counters": self.counters, "tick": self.tick}

    def _new(self, typ: str, value: str) -> str:
        n = self.counters.get(typ, 0) + 1
        self.counters[typ] = n
        key = f"{typ}_{n}"
        self.entities[key] = {"type": typ, "value": value, "variants": [], "count": 0, "last_seen": 0}
        return key

    def _resolve_person(self, value: str) -> str:
        toks = value.split()
        cands = []
        for key, ent in self.entities.items():
            if ent["type"] != "PERSON":
                continue
            etoks = ent["value"].split()
            e_first, e_last = etoks[:-1], etoks[-1]
            if len(toks) == 1:
                t = toks[0].casefold()
                if (t == e_last.casefold() or t in (f.casefold() for f in e_first)
                        or any(t == v.casefold() for v in ent["variants"])):
                    cands.append(key)
            else:
                first, last = toks[:-1], toks[-1]
                if last.casefold() != e_last.casefold():
                    if len(etoks) == 1 and etoks[0].casefold() == first[0].casefold():
                        cands.append(key)       # "Sarah" -> "Sarah Connor"
                    continue
                if not e_first or _first_compatible(first, e_first):
                    cands.append(key)
        if not cands:
            return self._new("PERSON", value)
        if len(cands) > 1:
            self.ambiguous += 1
        return max(cands, key=lambda k: self.entities[k]["last_seen"])

    def add_variant(self, key: str, value: str) -> None:
        """Weitere Schreibweise einer bereits bekannten Entität (aus der Propagation)."""
        value = " ".join(value.split())
        ent = self.entities[key]
        self.tick += 1
        if value not in ent["variants"]:
            ent["variants"].append(value)
        if ent["type"] != "PERSON":
            self.index[(ent["type"], _norm(ent["type"], value))] = key
        ent["count"] += 1
        ent["last_seen"] = self.tick

    def resolve(self, typ: str, value: str) -> str:
        value = " ".join(value.split())
        self.tick += 1
        if typ == "PERSON":
            key = self.index.get((typ, _norm(typ, value))) or self._resolve_person(value)
        else:
            key = self.index.get((typ, _norm(typ, value))) or self._new(typ, value)
        ent = self.entities[key]
        if value not in ent["variants"]:
            ent["variants"].append(value)
        self.index[(typ, _norm(typ, value))] = key
        ent["count"] += 1
        ent["last_seen"] = self.tick
        # Kanonische Form = längste Variante (vollständigster Name)
        if typ in ("PERSON", "ORG") and len(value) > len(ent["value"]):
            ent["value"] = value
        return key


_HON = r"(?:Herrn?|Frau|Hr\.|Fr\.|Mr\.?|Mrs\.?|Ms\.?|Dr\.|Prof\.)"


def propagate(text: str, spans: list[Span], linker: Linker, used_keys: set[str]) -> list[Span]:
    """Ersetzt weitere Vorkommen bereits bekannter Entitäten."""
    blocked = [(s.start, s.end) for s in spans]
    new: list[Span] = []
    for key, ent in linker.entities.items():
        local = key in used_keys
        for var in sorted(ent["variants"], key=len, reverse=True):
            # Werte aus früheren Dokumenten desselben Tresors nur, wenn sie spezifisch genug sind
            if len(var) < (3 if local else 6):
                continue
            for a, b in find_all(text, var, blocked):
                new.append(Span(a, b, ent["type"], "propagation", 0.6, rule="propagation", key=key))
                blocked.append((a, b))
        if ent["type"] == "PERSON" and local:
            toks = ent["value"].split()
            if len(toks) >= 2:
                rx = re.compile(rf"{_HON}\s+(?:(?:Dr\.|Prof\.)\s+)?({re.escape(toks[-1])})(?![\wäöüß])")
                for m in rx.finditer(text):
                    a, b = m.span(1)
                    if not any(a < y and x < b for x, y in blocked):
                        new.append(Span(a, b, "PERSON", "propagation", 0.6, rule="propagation:surname", key=key))
                        blocked.append((a, b))
    return new
