"""Regelbasierte Erkennung strukturierter personenbezogener Daten.

Muster + Prüfsummen + Kontextwörter (Context-Boosting wie in Presidio). Diese Stufe
ist deterministisch und präzise; sie liefert die "Anker", die bei Überlappungen
Vorrang vor modellbasierten Treffern haben (vgl. LLM-Redactor: Regex für
strukturierte Geheimnisse, NER/LLM für Namen und Kontext).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .common import Span
from .validators import (IBAN_LENGTHS, iban_valid, ipv4_valid, luhn_valid,
                         steuer_id_valid, svnr_valid)

Extract = Callable[[re.Match, str], list]


def _ctx(words) -> re.Pattern:
    return re.compile("|".join(r"(?<![a-zäöüß])" + re.escape(w.lower()) for w in words))


@dataclass
class Rule:
    name: str
    type: str
    rx: re.Pattern
    score: float = 0.9
    group: int | str = 0
    validator: Callable[[str], bool] | None = None
    refine: Callable[[str, int, int], tuple[int, int, bool]] | None = None
    extract: Extract | None = None
    ctx: re.Pattern | None = None
    ctx_mode: str = "none"      # none | required | if_invalid
    before: int = 40
    after: int = 20
    postcheck: Callable[[str, str, int, int], bool] | None = None
    tags: dict = field(default_factory=dict)


def _has_ctx(text: str, s: int, e: int, rule: Rule) -> bool:
    if rule.ctx is None:
        return False
    window = text[max(0, s - rule.before):s].lower() + "   " + text[e:e + rule.after].lower()
    return bool(rule.ctx.search(window))


# --------------------------------------------------------------------------- #
# Hilfsfunktionen für einzelne Regeln
# --------------------------------------------------------------------------- #
def _refine_iban(text: str, s: int, e: int) -> tuple[int, int, bool]:
    val = text[s:e]
    parts = val.split(" ")
    for k in range(len(parts), 1, -1):
        cand = " ".join(parts[:k])
        if iban_valid(cand):
            return s, s + len(cand), True
    compact = val.replace(" ", "")
    length = IBAN_LENGTHS.get(compact[:2])
    if " " not in val and length and len(compact) > length and iban_valid(compact[:length]):
        return s, s + length, True
    return s, e, False


def _card_valid(val: str) -> bool:
    d = re.sub(r"\D", "", val)
    return luhn_valid(d) and d[0] in "23456"


def _phone_check(val: str, text: str, s: int, e: int) -> bool:
    n = sum(c.isdigit() for c in val)
    if not 7 <= n <= 15:
        return False
    if re.fullmatch(r"\d{1,2}[/.]\d{1,2}[/.]\d{2,4}", val.strip()):   # Datum, keine Nummer
        return False
    after = text[e:e + 6].lower()
    if re.match(r"\s?(uhr|h\b|%|euro|eur|€|mio|stück)", after):
        return False
    return True


_STOP_SMALL = {
    "von", "der", "die", "das", "mit", "the", "from", "for", "und", "and", "ist", "is",
    "nicht", "not", "über", "via", "bei", "zum", "zur", "fehlgeschlagen", "failed", "an",
    "auf", "für", "im", "in", "zu", "per", "seit", "page", "seite", "daten",
}


def _extract_password(m: re.Match, text: str) -> list:
    s, e = m.span("val")
    if m.group("q"):
        return [(s, e)]
    # Ohne Anführungszeichen: Satzzeichen am Ende gehören (fast immer) zum Satz.
    while e > s and text[e - 1] in ".,;:)?!":
        e -= 1
    val = text[s:e]
    if len(val) < 3 or val.lower() in _STOP_SMALL:
        return []
    if not re.search(r"[\d\W_]", val):      # Ziffer oder Sonderzeichen nötig
        return []
    return [(s, e)]


def _extract_username(m: re.Match, text: str) -> list:
    s, e = m.span("val")
    while e > s and text[e - 1] in ".,;:-":
        e -= 1
    val = text[s:e]
    if len(val) < 3 or val.lower() in _STOP_SMALL:
        return []
    if m.group("q") or m.group("conn") or re.search(r"[\d._@\-]", val):
        return [(s, e)]
    return []


_ID_KW_EXCLUDE = re.compile(
    r"^(?:ticket|vorgang|case|issue|bug|telefon|tel|handy|mobil|fax|haus|steuer|"
    r"sozialversicherung|kfz|kenn|seiten|seite|page|zimmer|raum|gleis|lini|bus|zug|"
    r"version|ver|artikel|art|paragraph|abs|kapitel|folge|lauf|serien|nummer$|nr$|no$|id$)",
    re.I)
_ID_TOKEN = re.compile(
    r"(?<![\w\-#])([A-Z0-9][A-Za-z0-9]*(?:(?:[\-/.]|\s(?=\d))[A-Za-z0-9]+)*)")


def _extract_contextual_id(m: re.Match, text: str) -> list:
    kw = m.group("kw")
    prefix = kw.lower().replace("-", "")
    # "Nummer"/"Nr." ohne Präfix ist erlaubt (z. B. "Vertrag mit der Nummer ..."),
    # Präfixe wie Ticket-/Telefon-/Hausnummer werden ausgeschlossen.
    stem = re.sub(r"(nummer|nr\.?|number|no\.?|id|kennung|aktenzeichen|az\.?)$", "", prefix)
    if stem and _ID_KW_EXCLUDE.match(stem):
        return []
    s0 = m.end()
    window = text[s0:s0 + 45]
    stop = re.search(r"[.!?;]\s|\n", window)
    if stop:
        window = window[:stop.start() + 1]
    for t in _ID_TOKEN.finditer(window):
        tok = t.group(1).rstrip(".")
        digits = sum(c.isdigit() for c in tok)
        if digits >= 3 and len(tok) >= 4 and not re.fullmatch(r"\d{1,2}\.\d{1,2}\.\d{2,4}", tok):
            if re.fullmatch(r"(19|20)\d{2}", tok):      # Jahreszahl
                continue
            a = s0 + t.start(1)
            return [(a, a + len(tok))]
    return []


# --------------------------------------------------------------------------- #
# Muster
# --------------------------------------------------------------------------- #
_SUF = r"(?:straße|strasse|str\.|gasse|weg|allee|platz|ring|damm|ufer|chaussee|steig|pfad|markt|hof|anger|zeile|promenade|kai|stieg|wall|graben|berg)"
_CITY = r"[A-ZÄÖÜ][a-zäöüß]+(?:-[A-ZÄÖÜ][a-zäöüß]+)*(?:\s(?:am|an\sder|im|ob\sder)\s[A-ZÄÖÜ][a-zäöüß]+)?"
_STREET1 = rf"[A-ZÄÖÜ][a-zäöüß]*(?:-[A-ZÄÖÜ]?[a-zäöüß]+)*-?(?i:{_SUF})"
_STREET2 = r"[A-ZÄÖÜ][a-zäöüß]+(?:er|sche|scher)\s(?i:straße|strasse|str\.|weg|allee|platz|ring|damm|gasse|chaussee|ufer)"
_HNR = r"\s\d{1,4}(?:\s?[a-zA-Z](?![\wäöüß]))?(?:\s?[-–/]\s?\d{1,4}[a-zA-Z]?)?"
_TAIL = rf"(?:,?\s(?:D-)?\d{{5}}\s{_CITY}|\s(?:in|in\sder|bei)\s{_CITY})?"
_MONTHS = r"(?:Jan(?:uar)?|Feb(?:ruar)?|März|Mär|Apr(?:il)?|Mai|Juni?|Juli?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Okt(?:ober)?|Nov(?:ember)?|Dez(?:ember)?)"
_MONTHS_EN = r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
_DATE = (r"(?<!\d)(?:\d{1,2}\.\s?\d{1,2}\.\s?(?:19|20)?\d{2}"
         rf"|\d{{1,2}}\.\s?{_MONTHS}\.?\s(?:19|20)\d{{2}}"
         r"|\d{1,2}/\d{1,2}/(?:19|20)\d{2}|(?:19|20)\d{2}-\d{2}-\d{2}"
         rf"|{_MONTHS_EN}\s\d{{1,2}},?\s(?:19|20)\d{{2}}|\d{{1,2}}\s{_MONTHS_EN}\s(?:19|20)\d{{2}})(?!\d)")

CARD_CTX = _ctx(["visa", "mastercard", "master card", "amex", "american express", "kreditkarte",
                 "karte", "kartennummer", "credit card", "card", "maestro", "diners"])
IBAN_CTX = _ctx(["iban", "konto", "bankverbindung", "account", "überweis"])
STEUER_CTX = _ctx(["steuer-id", "steuerid", "steuer-identifikationsnummer", "identifikationsnummer",
                   "idnr", "steuerliche", "tin", "tax id", "steuer id"])
SVNR_CTX = _ctx(["sozialversicherung", "rentenversicherung", "sv-nummer", "svnr", "sv-nr",
                 "versicherungsnummer", "rvnr", "social security"])
KV_CTX = _ctx(["versicherten", "krankenversicherung", "krankenkasse", "kv-nummer", "kvnr",
               "versicherungsnummer", "health insurance"])
BIRTH_CTX = _ctx(["geb.", "geboren", "geburt", "date of birth", "dob", "born", "birth"])
SALARY_CTX = _ctx(["gehalt", "lohn", "einkommen", "verdienst", "salary", "vergütung", "bezüge",
                   "rente", "stundenlohn", "income", "wage"])
SECRET_CTX = _ctx(["api-key", "api key", "apikey", "api-schlüssel", "token", "secret", "access key",
                   "bearer", "private key", "client-secret", "zugangsschlüssel"])
PLATE_CTX = _ctx(["kennzeichen", "kfz", "nummernschild", "license plate", "fahrzeug"])
PHONE_CTX = _ctx(["tel", "phone", "call", "fon", "mobil", "handy", "fax", "durchwahl", "anruf"])

RULES: list[Rule] = [
    # --- Kontakt -------------------------------------------------------------
    Rule("email", "EMAIL", re.compile(
        r"(?<![\w.+\-])[A-Za-z0-9][A-Za-z0-9._%+\-]*@[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?"
        r"(?:\.[A-Za-z0-9](?:[A-Za-z0-9\-]*[A-Za-z0-9])?)*\.[A-Za-z]{2,24}\b"), 0.98),
    Rule("phone", "TELEFON", re.compile(
        r"(?<![\w+\-/.,])(?:(?:\+|00)\d{1,3}[ \-]?(?:\(0\)[ \-]?)?\d{1,5}|\(?0\d{1,5}\)?)"
        r"(?:[ \-/]{1,3}\d{1,9}){0,5}(?![\w/]|[.,]\d)"), 0.85, postcheck=_phone_check),
    Rule("phone_us", "TELEFON", re.compile(r"(?<!\w)\(\d{3}\)\s?\d{3}[\-\s]\d{4}(?!\w)"), 0.8),
    Rule("phone_dash", "TELEFON", re.compile(r"(?<![\w\-])\d{3}-\d{3}-\d{4}(?![\w\-])"), 0.75,
         ctx=PHONE_CTX, ctx_mode="required"),
    # --- Finanzen ------------------------------------------------------------
    Rule("iban", "IBAN", re.compile(r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b"),
         0.97, refine=_refine_iban, ctx=IBAN_CTX, ctx_mode="if_invalid"),
    Rule("card", "KARTE", re.compile(r"(?<![\d+\-])(?:\d[ \-]?){12,18}\d(?![\d\-])"), 0.9,
         validator=_card_valid, ctx=CARD_CTX, ctx_mode="if_invalid", before=25),
    Rule("salary", "FINANZ", re.compile(
        r"(?<![\d.,])(?:\d{1,3}(?:[.\s]\d{3})+|\d+)(?:,\d{1,2})?\s?(?:Euro|EUR|€)(?:\s(?:brutto|netto))?"
        r"|(?:€|EUR)\s?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?"), 0.8,
         ctx=SALARY_CTX, ctx_mode="required", before=60),
    # --- Kennnummern ---------------------------------------------------------
    Rule("steuer_id", "ID", re.compile(r"(?<![\d\-/])\d{2} ?\d{3} ?\d{3} ?\d{3}(?![\d\-/])"), 0.9,
         validator=steuer_id_valid, ctx=STEUER_CTX, ctx_mode="if_invalid"),
    Rule("steuernummer", "ID", re.compile(r"(?<![\d/])\d{2,3}/\d{3,4}/\d{4,5}(?![\d/])"), 0.75),
    Rule("svnr", "ID", re.compile(r"(?<!\w)\d{2} ?\d{6} ?[A-Z] ?\d{3}(?!\w)"), 0.9,
         validator=svnr_valid, ctx=SVNR_CTX, ctx_mode="if_invalid"),
    Rule("kvnr", "ID", re.compile(r"\b[A-Z]\d{9}\b"), 0.85, ctx=KV_CTX, ctx_mode="required", before=60),
    Rule("ust_id", "ID", re.compile(r"\bDE ?\d{9}\b"), 0.85),
    Rule("handelsregister", "ID", re.compile(r"\bHR[AB] ?\d{1,6}(?: ?[A-Z]{1,2}\b)?"), 0.85),
    Rule("structured_id", "ID", re.compile(r"(?<![\w\-])[A-Z]{1,4}-\d{2,4}-\d{3,}(?![\w\-])"), 0.65),
    Rule("contextual_id", "ID", re.compile(
        r"(?i)(?<![a-zäöüß])(?P<kw>[a-zäöüß\-]*(?:nummer|nr\.|nr|number|no\.|kennung|aktenzeichen|az\.|\bid))(?![a-zäöüß])"),
         0.7, extract=_extract_contextual_id),
    # --- Netzwerk ------------------------------------------------------------
    Rule("ipv4", "NETZ", re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?!\d|\.\d)"), 0.9, validator=ipv4_valid),
    Rule("ipv6", "NETZ", re.compile(r"(?<![:\w])(?:[0-9A-Fa-f]{1,4}:){4,7}[0-9A-Fa-f]{1,4}(?![:\w])"), 0.85),
    Rule("ipv6_short", "NETZ", re.compile(r"(?<![:\w])[0-9A-Fa-f]{1,4}(?::[0-9A-Fa-f]{1,4})*::(?:[0-9A-Fa-f]{1,4}:)*[0-9A-Fa-f]{1,4}(?![:\w])"), 0.75),
    Rule("internal_host", "NETZ", re.compile(
        r"(?<![\w.\-@/])[a-z0-9][a-z0-9\-]*(?:\.[a-z0-9][a-z0-9\-]*)*\."
        r"(?:intern|internal|local|lan|corp|localdomain|intranet)(?:\.[a-z0-9\-]+)*\b"), 0.85),
    # --- Zugangsdaten ----------------------------------------------------------
    Rule("password", "CREDENTIAL", re.compile(
        r"(?i)(?<![a-zäöüß])(?:passwort|kennwort|password|passwd|pwd|pw|pin|zugangscode|passcode)(?![a-zäöüß])"
        r"\s*(?:(?:ist|lautet|lautete|is|was|war)\s+)?(?:[:=]\s*)?(?P<q>[\"'„“‚«»])?(?P<val>[^\s\"'“”„‚«»]{3,64})"),
         0.95, extract=_extract_password),
    Rule("username", "CREDENTIAL", re.compile(
        r"(?i)(?<![a-zäöüß])(?:benutzername|benutzerkennung|nutzername|username|user ?name|user-?id|"
        r"login(?:name)?|anmeldename|account(?:name)?)(?![a-zäöüß])"
        r"\s*(?P<conn>(?:ist|lautet|lautete|is|was|war)\s+|[:=]\s*)?(?P<q>[\"'„“‚«»])?"
        r"(?P<val>[A-Za-z0-9][A-Za-z0-9._\-@]{2,40})"), 0.9, extract=_extract_username),
    Rule("api_key_known", "CREDENTIAL", re.compile(
        r"\b(?:(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{10,}|sk-[A-Za-z0-9_\-]{20,}|gh[pousr]_[A-Za-z0-9]{30,}"
        r"|AKIA[0-9A-Z]{16}|xox[abprs]-[A-Za-z0-9\-]{10,}|AIza[0-9A-Za-z_\-]{35}"
        r"|eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,})\b"), 0.98),
    Rule("private_key", "CREDENTIAL", re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----"), 0.99),
    Rule("api_key_ctx", "CREDENTIAL", re.compile(r"(?<![\w\-])[A-Za-z0-9_\-]{20,}(?![\w\-])"), 0.8,
         validator=lambda v: bool(re.search(r"\d", v) and re.search(r"[A-Za-z]", v)),
         ctx=SECRET_CTX, ctx_mode="required"),
    # --- Fahrzeug ----------------------------------------------------------------
    Rule("plate", "KFZ", re.compile(r"(?<![\w\-])[A-ZÄÖÜ]{1,3}-[A-Z]{1,2} ?\d{1,4}[EH]?(?![\w\-])"), 0.8),
    Rule("plate_space", "KFZ", re.compile(r"(?<![\w\-])[A-ZÄÖÜ]{1,3} [A-Z]{1,2} \d{1,4}[EH]?(?![\w\-])"), 0.75,
         ctx=PLATE_CTX, ctx_mode="required"),
    # --- Geburtsdatum --------------------------------------------------------------
    Rule("birthdate", "DATUM", re.compile(_DATE), 0.9, ctx=BIRTH_CTX, ctx_mode="required", before=30, after=15),
    # --- Gesundheits-/Statusbegriffe (Art. 9 DSGVO), Lexikon ------------------------
    # Ergänzt das LLM bei Begriffen, die es nicht zuverlässig als Gesundheitsdatum meldet.
    Rule("health_lexicon", "GESUNDHEIT", re.compile(
        r"(?i)(?<![\wäöüß])(?:krankgeschrieben|krank\s?gemeldet|krankgemeldet|krankschreibung|krankmeldung|"
        r"arbeitsunfähig(?:keit)?(?:sbescheinigung)?|au-bescheinigung|reha-?(?:maßnahme|klinik|aufenthalt)|"
        r"rehabilitation|schwanger(?:schaft)?|mutterschutz|schwerbehindert(?:en)?(?:ausweis)?|"
        r"schwerbehinderung|pflegegrad\s?\d?|chemotherapie|psychotherapie|dialyse|"
        r"on\ssick\sleave|sick\sleave|pregnan(?:t|cy)|maternity\sleave)(?![\wäöüß])"), 0.75),
    # --- Adressen --------------------------------------------------------------------
    Rule("address_de", "ADRESSE", re.compile(
        rf"(?<![\wäöüÄÖÜß\-])(?:{_STREET2}|{_STREET1}){_HNR}{_TAIL}"), 0.85),
    Rule("postbox", "ADRESSE", re.compile(rf"\b(?:Postfach|PF)\s\d{{2,7}}(?:,?\s\d{{5}}\s{_CITY})?"), 0.85),
    Rule("plz_city", "ADRESSE", re.compile(rf"(?:(?<=,\s)|(?<=\bin\s)|(?<=^))(?:D-)?\d{{5}}\s{_CITY}", re.M), 0.6),
    Rule("address_en", "ADRESSE", re.compile(
        r"\b\d{1,5}[A-Z]?\s(?:[A-Z][a-z]+\s){1,3}(?:Street|St\.|Road|Rd\.|Avenue|Ave\.|Lane|Ln\.|Drive|"
        r"Boulevard|Blvd\.|Way|Court|Ct\.|Place|Pl\.|Square|Terrace|Close|Crescent)"
        r"(?:,?\s[A-Z][a-z]+(?:,?\s[A-Z]{1,2}\d[A-Z\d]?\s?\d[A-Z]{2})?)?"), 0.85),
    Rule("uk_postcode", "ADRESSE", re.compile(r"\b[A-Z]{1,2}\d[A-Z\d]?\s\d[A-Z]{2}\b"), 0.6),
]


def detect_regex(text: str, rules: list[Rule] | None = None) -> list[Span]:
    spans: list[Span] = []
    for rule in rules or RULES:
        for m in rule.rx.finditer(text):
            ranges = rule.extract(m, text) if rule.extract else [m.span(rule.group)]
            for s, e in ranges:
                if e <= s:
                    continue
                valid = True
                if rule.refine:
                    s, e, valid = rule.refine(text, s, e)
                elif rule.validator:
                    valid = rule.validator(text[s:e])
                if rule.postcheck and not rule.postcheck(text[s:e], text, s, e):
                    continue
                ctx = _has_ctx(text, s, e, rule) if rule.ctx_mode != "none" else False
                if rule.ctx_mode == "required" and not ctx:
                    continue
                if not valid:
                    if rule.ctx_mode in ("if_invalid", "required") and ctx:
                        score = rule.score - 0.15
                    else:
                        continue
                else:
                    score = rule.score
                spans.append(Span(s, e, rule.type, "regex", score, rule=rule.name))
    return spans
