"""Hybride Erkennungs- und Ersetzungspipeline.

Ablauf (pro Dokument):
  1. Regex + Prüfsummen + Kontextwörter          -> präzise Anker für strukturierte Daten
  2. GLiNER Zero-Shot-NER                         -> Namen, Organisationen, Orte, Diagnosen
  3. Lokales LLM (Ollama) mit Kandidaten-Hinweisen -> Kontextentscheidungen, Ergänzungen
  4. Fusion + Überlappungsauflösung (Regex-Anker > längster Treffer)
  5. Entitätsverknüpfung + Propagation            -> konsistente Platzhalter
  6. Residual-Runde(n): LLM prüft den pseudonymisierten Text als Gegenspieler
  7. Ersetzung durch typisierte Platzhalter [TYP_n]
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from .common import (SOURCE_PRIORITY, Span, alnum_count, chunk_spans, eprint, locate,
                     overlaps_any, trim_span)
from .linking import Linker, propagate
from .regex_rules import detect_regex

HARD_NER_TYPES = {"EMAIL", "TELEFON", "IBAN", "KARTE", "NETZ", "KFZ"}
LONG_OK_TYPES = {"GESUNDHEIT", "MERKMAL", "ADRESSE", "GESCHAEFTSZAHL", "FINANZ", "DATUM", "ORG"}
PLACEHOLDER_IN_TEXT = re.compile(r"[\[<⟦⟨{«]\s*[A-ZÄÖÜa-zäöü]+[ _\-]?\d+\s*[\]>⟧⟩}»]")
_BARE_WORDS = {
    "herr", "frau", "dr", "dr.", "prof", "prof.", "er", "sie", "ihn", "ihm", "ihr", "ihre", "sein",
    "seine", "he", "she", "him", "her", "his", "they", "kunde", "kundin", "patient", "patientin",
}


@dataclass
class Result:
    text: str
    spans: list[Span]
    stats: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    surfaces: list[tuple[str, str]] = field(default_factory=list)   # (key, Originalschreibweise)


def resolve_overlaps(spans: list[Span]) -> list[Span]:
    """Clusterweise Auswahl: Regex-Treffer zuerst (präzise Grenzen), dann längster Treffer.
    Überstehende Reste längerer Modell-Treffer bleiben erhalten, wenn sie substanziell sind."""
    spans = sorted(spans, key=lambda s: (s.start, -s.end))
    clusters, cur, cur_end = [], [], -1
    for s in spans:
        if cur and s.start < cur_end:
            cur.append(s)
            cur_end = max(cur_end, s.end)
        else:
            if cur:
                clusters.append(cur)
            cur, cur_end = [s], s.end
    if cur:
        clusters.append(cur)

    chosen: list[Span] = []
    for cl in clusters:
        regex = sorted([s for s in cl if s.source == "regex"], key=lambda s: (-s.length, -s.score))
        other = sorted([s for s in cl if s.source != "regex"],
                       key=lambda s: (-s.length, -SOURCE_PRIORITY.get(s.source, 0), -s.score))
        picked: list[Span] = []
        for s in regex:
            if not any(s.overlaps(p) for p in picked):
                picked.append(s)
        for s in other:
            if not any(s.overlaps(p) for p in picked):
                picked.append(s)
        chosen.extend(picked)
    return sorted(chosen, key=lambda s: s.start)


def _clip_remainders(text: str, spans: list[Span], chosen: list[Span]) -> list[Span]:
    """Teile von verworfenen Modell-Treffern, die außerhalb aller gewählten Spans liegen
    (z. B. Firmenname vor einer per Regex erkannten Adresse), werden nachgezogen."""
    extra = []
    taken = [(c.start, c.end) for c in chosen]
    for s in spans:
        if s.source == "regex" or any(s is c for c in chosen):
            continue
        pieces = [(s.start, s.end)]
        for a, b in sorted(taken):
            nxt = []
            for x, y in pieces:
                if b <= x or a >= y:
                    nxt.append((x, y))
                else:
                    if x < a:
                        nxt.append((x, a))
                    if b < y:
                        nxt.append((b, y))
            pieces = nxt
        if pieces == [(s.start, s.end)]:
            continue
        for x, y in pieces:
            x, y = trim_span(text, x, y, s.type)
            if y > x and alnum_count(text[x:y]) >= 6 and not overlaps_any(x, y, taken):
                extra.append(Span(x, y, s.type, s.source, s.score * 0.9, rule=s.rule + ":clip"))
                taken.append((x, y))
    return extra


class Pseudonymizer:
    def __init__(self, cfg: dict, use_ner: bool = True, use_llm: bool = True,
                 require_llm: bool = False, verbose: bool = False):
        self.cfg = cfg
        self.verbose = verbose
        self.init_warnings: list[str] = []
        self.ner = None
        self.llm = None
        ner_cfg, llm_cfg = cfg["ner"], cfg["llm"]
        if use_ner and ner_cfg.get("enabled", True):
            try:
                from .ner import NerDetector
                t = time.time()
                self.ner = NerDetector(ner_cfg["model"], ner_cfg.get("labels"), ner_cfg.get("threshold", 0.5),
                                       ner_cfg.get("max_chunk_chars", 1000))
                self._log(f"NER geladen ({ner_cfg['model']}, {time.time() - t:.1f}s)")
            except Exception as ex:  # noqa: BLE001
                self.init_warnings.append(f"NER nicht verfügbar ({type(ex).__name__}: {ex}); weiter ohne NER.")
        if use_llm and llm_cfg.get("enabled", True):
            from .llm import LLMUnavailable, OllamaLLM
            llm = OllamaLLM(llm_cfg["host"], llm_cfg["model"], llm_cfg.get("think", False),
                            llm_cfg.get("temperature", 0.0), llm_cfg.get("seed", 42),
                            llm_cfg.get("num_ctx", 8192), llm_cfg.get("timeout", 300))
            try:
                llm.check()
                self.llm = llm
            except LLMUnavailable as ex:
                if require_llm or llm_cfg.get("required", False):
                    raise
                self.init_warnings.append(f"LLM nicht verfügbar ({ex}); weiter ohne LLM-Stufe "
                                          "(geringere Erkennung von Kontextfällen!).")
        self.disabled = set(cfg.get("types_disabled", []))

    def _log(self, msg: str) -> None:
        if self.verbose:
            eprint("[pseudo]", msg)

    # ------------------------------------------------------------------ Stufen
    def _llm_spans(self, text: str, candidates: list[Span], stats: dict) -> list[Span]:
        out: list[Span] = []
        for cs, ce in chunk_spans(text, self.cfg["llm"].get("max_chunk_chars", 2500)):
            chunk = text[cs:ce]
            seen, hints = set(), []
            for c in candidates:
                if cs <= c.start and c.end <= ce:
                    h = (text[c.start:c.end], c.type)
                    if h not in seen:
                        seen.add(h)
                        hints.append({"text": h[0], "type": h[1]})
            items = self.llm.extract(chunk, hints)
            out.extend(self._place(text, items, stats, "llm", offset=cs, chunk=chunk))
        return out

    def _place(self, text: str, items: list[dict], stats: dict, source: str, offset: int = 0,
               chunk: str | None = None, blocked=()) -> list[Span]:
        out = []
        hay = chunk if chunk is not None else text
        for it in items:
            needle, typ = it["text"].strip(), it["type"]
            if (len(needle) < 2 or needle.casefold() in _BARE_WORDS
                    or PLACEHOLDER_IN_TEXT.search(needle)):
                stats["llm_rejected"] = stats.get("llm_rejected", 0) + 1
                continue
            limit = 120 if typ in LONG_OK_TYPES else 60
            if len(needle) > limit:
                stats["llm_rejected"] = stats.get("llm_rejected", 0) + 1
                continue
            hits = locate(hay, needle, [(a - offset, b - offset) for a, b in blocked])
            if not hits:
                stats["llm_not_found"] = stats.get("llm_not_found", 0) + 1
                continue
            for a, b in hits:
                out.append(Span(a + offset, b + offset, typ, source, 0.8, rule=f"{source}:{typ}"))
        return out

    def _filter_ner(self, ner_spans: list[Span], llm_spans: list[Span]) -> list[Span]:
        ner_cfg = self.cfg["ner"]
        keep = []
        for s in ner_spans:
            if self.llm is not None:
                confirmed = any(s.overlaps(l) for l in llm_spans)
                if (confirmed or (s.type in HARD_NER_TYPES and s.score >= ner_cfg.get("threshold", 0.5))
                        or (s.type == "PERSON" and s.score >= ner_cfg.get("person_accept_threshold", 0.85))):
                    keep.append(s)
            elif s.type != "ORT":
                keep.append(s)
        return keep

    def _finalize(self, text: str, spans: list[Span]) -> list[Span]:
        out = []
        for s in spans:
            if s.type in self.disabled:
                continue
            a, b = trim_span(text, s.start, s.end, s.type)
            if b <= a or alnum_count(text[a:b]) == 0:
                continue
            s.start, s.end = a, b
            out.append(s)
        chosen = resolve_overlaps(out)
        chosen += _clip_remainders(text, out, chosen)
        return sorted(chosen, key=lambda s: s.start)

    def _assign(self, text: str, spans: list[Span], linker: Linker, used: set[str]) -> list[Span]:
        for s in spans:
            if s.key is None:
                s.key = linker.resolve(s.type, text[s.start:s.end])
            used.add(s.key)
        prop = propagate(text, spans, linker, used)
        for p in prop:
            linker.add_variant(p.key, text[p.start:p.end])
        return sorted(spans + prop, key=lambda s: s.start)

    # ------------------------------------------------------------------- Run
    def run(self, text: str, linker: Linker, mode: str = "pseudo") -> Result:
        t0 = time.time()
        stats: dict = {"llm_not_found": 0, "llm_rejected": 0}
        warnings = list(self.init_warnings)

        regex_spans = detect_regex(text)
        ner_spans = self.ner.detect(text) if self.ner else []
        llm_spans: list[Span] = []
        if self.llm:
            llm_spans = self._llm_spans(text, regex_spans + ner_spans, stats)
        fused = regex_spans + llm_spans + self._filter_ner(ner_spans, llm_spans)
        stats["candidates"] = {"regex": len(regex_spans), "ner": len(ner_spans), "llm": len(llm_spans)}

        used: set[str] = set()
        spans = self._assign(text, self._finalize(text, fused), linker, used)

        # Residual-Runden: LLM als Gegenspieler auf dem pseudonymisierten Text
        stats["residual_added"] = 0
        if self.llm:
            for _ in range(int(self.cfg["llm"].get("residual_rounds", 1))):
                pseudo = render(text, spans, self.cfg["placeholder_format"])
                found: list[Span] = []
                blocked = [(s.start, s.end) for s in spans]
                for cs, ce in chunk_spans(pseudo, self.cfg["llm"].get("max_chunk_chars", 2500)):
                    items = self.llm.extract(pseudo[cs:ce], residual=True)
                    found += self._place(text, items, stats, "residual", blocked=blocked)
                found = [f for f in self._finalize(text, found) if not overlaps_any(f.start, f.end, blocked)]
                if not found:
                    break
                stats["residual_added"] += len(found)
                spans = self._assign(text, sorted(spans + found, key=lambda s: s.start), linker, used)

        fmt = self.cfg["placeholder_format"] if mode == "pseudo" else "[{type}]"
        out = render(text, spans, fmt)
        if linker.ambiguous:
            warnings.append(f"{linker.ambiguous} mehrdeutige Namenszuordnung(en) (gleicher Nachname, "
                            "mehrere Personen) – bitte prüfen.")
        by_type, by_source = {}, {}
        for s in spans:
            by_type[s.type] = by_type.get(s.type, 0) + 1
            by_source[s.source] = by_source.get(s.source, 0) + 1
        stats.update({"replacements": len(spans), "entities": len(used), "by_type": by_type,
                      "by_source": by_source, "llm_calls": self.llm.calls if self.llm else 0,
                      "llm_used": self.llm is not None, "ner_used": self.ner is not None,
                      "seconds": round(time.time() - t0, 2)})
        surfaces = [(s.key, text[s.start:s.end]) for s in sorted(spans, key=lambda s: s.start)]
        return Result(out, spans, stats, warnings, surfaces)


def render(text: str, spans: list[Span], fmt: str) -> str:
    parts, pos = [], 0
    for s in sorted(spans, key=lambda s: s.start):
        parts.append(text[pos:s.start])
        parts.append(fmt.format(key=s.key, type=s.type))
        pos = s.end
    parts.append(text[pos:])
    return "".join(parts)
