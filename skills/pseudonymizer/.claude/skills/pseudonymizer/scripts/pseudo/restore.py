"""Rückführung (Re-Identifikation) von Platzhaltern in Texten und LLM-Antworten.

Validierung nach PromptGraph (arXiv:2607.10709): Ein Platzhalter wird nur dann
ersetzt, wenn (1) sein Schlüssel im Tresor existiert und (2) der Typ zum Eintrag
passt. Erfundene oder veränderte Platzhalter bleiben unaufgelöst stehen und
werden gemeldet; nie wird "der ähnlichste" Wert geraten.

Tolerant gegenüber typischen LLM-Varianten: [PERSON_1], <PERSON_1>, ⟦PERSON_1⟧,
[[PERSON_1]], [Person_1], [PERSON 1], \\[PERSON_1\\] und PERSON_1 ohne Klammern.
"""
from __future__ import annotations

import hashlib
import re

from .common import HONORIFIC_BEFORE_RE, TYPES


def placeholder_regex(types=TYPES) -> re.Pattern:
    alt = "|".join(sorted(types, key=len, reverse=True))
    return re.compile(
        rf"\\?(?P<o>[\[<⟦⟨{{«]{{1,2}})\\?\s*(?P<t>{alt})[ _\-]?(?P<n>\d+)\s*\\?(?P<c>[\]>⟧⟩}}»]{{1,2}})"
        rf"|(?<![\w\[<⟦⟨{{«])(?P<bt>{alt})_(?P<bn>\d+)(?![\w\]>⟧⟩}}»])",
        re.IGNORECASE)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def restore_document(text: str, doc: dict, fmt: str) -> tuple[str, dict] | None:
    """Exakte Rückführung eines unveränderten, selbst erzeugten Dokuments:
    jedes Platzhalter-Vorkommen erhält seine Original-Schreibweise zurück."""
    if doc.get("sha256") != text_hash(text):
        return None
    open_, close = fmt.split("{key}")
    rx = re.compile(re.escape(open_) + r"([A-Z]+_\d+)" + re.escape(close))
    surfaces = doc.get("surfaces", [])
    matches = list(rx.finditer(text))
    if len(matches) != len(surfaces):
        return None
    parts, pos = [], 0
    for m, (key, surface) in zip(matches, surfaces):
        if m.group(1) != key:
            return None
        parts.append(text[pos:m.start()])
        parts.append(surface)
        pos = m.end()
    parts.append(text[pos:])
    keys = sorted({k for k, _ in surfaces})
    return "".join(parts), {"mode": "exakt", "placeholders_found": len(matches), "restored": len(matches),
                            "unresolved": [], "keys_used": keys, "keys_unused": []}


def _surname_variant_seen(ent: dict) -> bool:
    toks = ent["value"].split()
    return len(toks) > 1 and any(v.casefold() == toks[-1].casefold() for v in ent.get("variants", []))


def restore_text(text: str, entities: dict) -> tuple[str, dict]:
    rx = placeholder_regex()
    report = {"mode": "heuristisch", "placeholders_found": 0, "restored": 0, "unresolved": [],
              "keys_used": set()}

    def repl(m: re.Match) -> str:
        typ = (m.group("t") or m.group("bt")).upper()
        num = m.group("n") or m.group("bn")
        key = f"{typ}_{int(num)}"
        report["placeholders_found"] += 1
        ent = entities.get(key)
        if ent is None or ent.get("type") != typ:
            report["unresolved"].append(m.group(0))
            return m.group(0)
        report["restored"] += 1
        report["keys_used"].add(key)
        value = ent["value"]
        # "Herr [PERSON_1]" -> "Herr Schneider" statt "Herr Harald Schneider" – nur wenn die
        # Nachnamen-Form im Original tatsächlich vorkam ("seine Frau Petra Ortmann" bleibt vollständig).
        if (typ == "PERSON" and HONORIFIC_BEFORE_RE.search(m.string[:m.start()])
                and _surname_variant_seen(ent)):
            value = value.split()[-1]
        return value

    restored = rx.sub(repl, text)
    report["keys_used"] = sorted(report["keys_used"])
    report["keys_unused"] = sorted(set(entities) - set(report["keys_used"]))
    return restored, report
