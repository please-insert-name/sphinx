"""Gemeinsame Datentypen und Text-Hilfsfunktionen."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass

# Kanonische Kategorien. Der Platzhalter lautet <KATEGORIE>_<n>, z. B. PERSON_1.
TYPES: tuple[str, ...] = (
    "PERSON",          # Namen natürlicher Personen
    "ORG",             # Firmen, Praxen, Vereine, nicht-öffentliche Organisationen
    "ADRESSE",         # Straße + Hausnummer (+ PLZ/Ort), Postfach
    "ORT",             # Orte mit Personenbezug (Geburts-/Wohnort)
    "DATUM",           # personenbezogene Daten (Geburtsdatum, Arzttermin ...)
    "TELEFON",
    "EMAIL",
    "IBAN",
    "KARTE",           # Kreditkartennummern
    "ID",              # Kennnummern (Steuer-ID, SV-Nr., Kunden-, Vertragsnummer ...)
    "GESUNDHEIT",      # Art. 9 DSGVO: Diagnosen, Behandlungen, Allergien ...
    "CREDENTIAL",      # Passwörter, Benutzernamen, API-Schlüssel, Tokens
    "NETZ",            # IP-Adressen, interne Hostnamen
    "KFZ",             # Kfz-Kennzeichen
    "FINANZ",          # Gehalt, Einkommen, Vermögen einer Person
    "GESCHAEFTSZAHL",  # vertrauliche Geschäftszahlen (Umsatz, Auftragsvolumen)
    "MERKMAL",         # Quasi-Identifikatoren / sensible Merkmale (Alter, Familienstand ...)
)

# Präzedenz bei Überlappungen (höher gewinnt bei gleicher Länge)
SOURCE_PRIORITY = {"regex": 4, "llm": 3, "residual": 3, "ner": 2, "propagation": 1}


@dataclass
class Span:
    start: int
    end: int
    type: str
    source: str
    score: float = 1.0
    rule: str = ""
    key: str | None = None

    @property
    def length(self) -> int:
        return self.end - self.start

    def overlaps(self, other: "Span") -> bool:
        return self.start < other.end and other.start < self.end


def eprint(*args) -> None:
    print(*args, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Anreden / Titel
# --------------------------------------------------------------------------- #
# Anreden und akademische Titel bleiben VOR dem Platzhalter stehen ("Herr [PERSON_1]").
# Das hält den Text grammatisch und erlaubt der Rückführung, "Herr <Nachname>" zu bilden.
HONORIFIC_PREFIX_RE = re.compile(
    r"(?:Herrn?|Frau|Hr\.|Fr\.|Fam\.|Familie"
    r"|Dr\.(?:\s?(?:med|rer\.\s?nat|jur|phil|med\.\s?dent|-Ing)\.)?"
    r"|Prof\.|Dipl\.-[A-Za-zÄÖÜäöü]+\.?"
    r"|Mr\.?|Mrs\.?|Ms\.?|Miss|Mx\.?|Sir|Dame)\s+"
)
ARTICLE_PREFIX_RE = re.compile(r"(?:der|die|den|dem|des|the|unser(?:e|en|em)?|ihr(?:e|en|em)?|sein(?:e|en|em)?)\s+", re.I)
HONORIFIC_BEFORE_RE = re.compile(r"(?:Herrn?|Frau|Mr\.?|Mrs\.?|Ms\.?)\s+$")


# --------------------------------------------------------------------------- #
# Suchen mit Wortgrenzen
# --------------------------------------------------------------------------- #
def _is_word(ch: str) -> bool:
    return ch.isalnum() or ch == "_"


def boundary_ok(text: str, i: int, j: int) -> bool:
    if i < j and _is_word(text[i]) and i > 0 and _is_word(text[i - 1]):
        return False
    if i < j and _is_word(text[j - 1]) and j < len(text) and _is_word(text[j]):
        return False
    return True


def overlaps_any(i: int, j: int, intervals) -> bool:
    return any(i < b and a < j for a, b in intervals)


def find_all(text: str, needle: str, blocked=(), case: bool = True) -> list[tuple[int, int]]:
    """Alle Vorkommen von needle (mit Wortgrenzen), die nicht mit blocked überlappen."""
    if not needle:
        return []
    out: list[tuple[int, int]] = []
    if case:
        start = 0
        while True:
            i = text.find(needle, start)
            if i < 0:
                break
            j = i + len(needle)
            if boundary_ok(text, i, j) and not overlaps_any(i, j, blocked):
                out.append((i, j))
            start = i + 1
        return out
    for m in re.finditer(re.escape(needle), text, re.I):
        i, j = m.span()
        if boundary_ok(text, i, j) and not overlaps_any(i, j, blocked):
            out.append((i, j))
    return out


def locate(text: str, needle: str, blocked=()) -> list[tuple[int, int]]:
    """Findet eine vom LLM gelieferte Zeichenkette im Text: exakt, dann
    leerzeichen-tolerant, dann ohne Groß-/Kleinschreibung."""
    needle = needle.strip()
    if not needle:
        return []
    hits = find_all(text, needle, blocked)
    if hits:
        return hits
    tokens = needle.split()
    if len(tokens) > 1:
        rx = re.compile(r"\s+".join(map(re.escape, tokens)))
        hits = [m.span() for m in rx.finditer(text)
                if boundary_ok(text, *m.span()) and not overlaps_any(*m.span(), blocked)]
        if hits:
            return hits
    return find_all(text, needle, blocked, case=False)


# --------------------------------------------------------------------------- #
# Segmentierung
# --------------------------------------------------------------------------- #
ABBREV = {
    "dr", "prof", "hr", "fr", "nr", "str", "geb", "ca", "bzw", "z.b", "u.a", "st", "mio",
    "mrd", "tel", "abs", "art", "vgl", "usw", "etc", "mr", "mrs", "ms", "jr", "sr", "no",
    "co", "inc", "ltd", "corp", "dipl", "ing", "med", "jur", "phil", "nat", "rer", "evtl",
    "ggf", "inkl", "zzgl", "max", "min", "bspw", "d.h", "e.v", "fa", "gmbh",
}


def paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans, pos = [], 0
    for m in re.finditer(r"\r?\n[ \t]*\r?\n", text):
        spans.append((pos, m.start()))
        pos = m.end()
    spans.append((pos, len(text)))
    out = []
    for s, e in spans:
        while s < e and text[s].isspace():
            s += 1
        while e > s and text[e - 1].isspace():
            e -= 1
        if e > s:
            out.append((s, e))
    return out


def sentence_spans(text: str, start: int, end: int) -> list[tuple[int, int]]:
    seg = text[start:end]
    cuts = [0]
    for m in re.finditer(r"[.!?]+[\"“”')\]]*\s+", seg):
        before = re.search(r"(\S+)$", seg[: m.start() + 1])
        tok = before.group(1).rstrip(".!?").lower() if before else ""
        if tok in ABBREV or re.fullmatch(r"\d{1,2}", tok) or len(tok) <= 1:
            continue
        cuts.append(m.end())
    cuts.append(len(seg))
    out = []
    for a, b in zip(cuts, cuts[1:]):
        if b > a and seg[a:b].strip():
            out.append((start + a, start + b))
    return out


def chunk_spans(text: str, max_chars: int) -> list[tuple[int, int]]:
    """Absätze; zu lange Absätze werden satzweise zu Blöcken <= max_chars gepackt."""
    chunks = []
    for ps, pe in paragraph_spans(text):
        if pe - ps <= max_chars:
            chunks.append((ps, pe))
            continue
        sents = sentence_spans(text, ps, pe) or [(ps, pe)]
        cur_s, cur_e = sents[0]
        for s, e in sents[1:]:
            if e - cur_s > max_chars:
                chunks.append((cur_s, cur_e))
                cur_s = s
            cur_e = e
        chunks.append((cur_s, cur_e))
    # Notfall: einzelne überlange Sätze hart an Leerzeichen trennen
    final = []
    for s, e in chunks:
        while e - s > max_chars * 1.5:
            cut = text.rfind(" ", s, s + max_chars)
            cut = cut if cut > s else s + max_chars
            final.append((s, cut))
            s = cut
        final.append((s, e))
    return final


# --------------------------------------------------------------------------- #
# Span-Bereinigung
# --------------------------------------------------------------------------- #
LEAD_CHARS = " \t\r\n\"'„“‚‘«»(<[{"
TRAIL_CHARS = " \t\r\n\"'“”‘’«»)>]},;:"
_KEEP_DOT = re.compile(r"(?:Str|Corp|Inc|Ltd|Co|e\.V|mbH|Bros|Nr|Jr|Sr|St)\.$")


def trim_span(text: str, s: int, e: int, typ: str) -> tuple[int, int]:
    if typ == "CREDENTIAL":
        while s < e and text[s] in " \t\r\n\"'„“‚‘«»":
            s += 1
        while e > s and text[e - 1] in " \t\r\n\"'“”‘’«»":
            e -= 1
        return s, e
    while s < e and text[s] in LEAD_CHARS:
        s += 1
    while e > s and text[e - 1] in TRAIL_CHARS:
        e -= 1
    while e > s and text[e - 1] in ".!?":
        if typ in ("ORG", "ADRESSE", "PERSON") and _KEEP_DOT.search(text[s:e]):
            break
        if typ == "DATUM" and re.search(r"(?:^|\D)\d{1,2}\.\d{1,2}\.$", text[s:e]):
            break    # "02.09." – der Punkt gehört zum Datum (Tag.Monat.)
        e -= 1
        while e > s and text[e - 1] in TRAIL_CHARS:
            e -= 1
    if typ == "PERSON":
        changed = True
        while changed:
            changed = False
            for rx in (HONORIFIC_PREFIX_RE, ARTICLE_PREFIX_RE):
                m = rx.match(text, s, e)
                if m and m.end() < e:
                    s = m.end()
                    changed = True
    return s, e


def alnum_count(s: str) -> int:
    return sum(1 for c in s if c.isalnum())
