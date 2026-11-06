"""Kontextuelle Erkennung über ein lokales LLM (Ollama).

Das LLM übernimmt, was Muster und NER schlecht können: Kontextentscheidungen
(Gesundheitsdaten, Quasi-Identifikatoren, "ist dieser Ort/dieses Datum
personenbezogen?"). Ansätze: LLM als Zero-/Few-Shot-Anonymisierer (Staab et al.,
ICLR 2024/2025), LLM-in-the-loop-Verifikation von NER-Kandidaten (Gunay et al.,
arXiv:2412.10918), Residual-Prüfung des pseudonymisierten Textes als
Gegenspieler (adversarial, vgl. Staab et al.; AURA arXiv:2605.30848).

Ausgabe wird per JSON-Schema (Ollama "format") erzwungen; die Temperatur ist 0.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

from .common import TYPES


class LLMUnavailable(RuntimeError):
    pass


SCHEMA = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {"type": "string", "enum": list(TYPES)},
                },
                "required": ["text", "type"],
            },
        }
    },
    "required": ["entities"],
}

POLICY = """Maskiert werden müssen (Kategorie in Klammern):
- (PERSON) Namen natürlicher Personen: Vor- und/oder Nachname, auch einzelne Vornamen in Anreden oder Erwähnungen. Anrede und Titel (Herr, Frau, Dr., Prof.) NICHT mit ausgeben – nur den Namen.
- (ORG) Namen von Firmen, Praxen, Kanzleien, Vereinen und sonstigen Organisationen, die Rückschlüsse auf Beteiligte erlauben (Kunden, Lieferanten, Arbeitgeber). NICHT: allgemein bekannte Marken, Plattformen und Dienstleister, die nur beiläufig als Mittel erwähnt werden (z. B. Zahlungsdienste, Kartenmarken, Suchmaschinen, Verkehrsunternehmen).
- (ADRESSE) Straße mit Hausnummer, ggf. mit PLZ und Ort; Postfächer.
- (ORT) Orte NUR, wenn sie zusammen mit einer Person etwas über sie verraten (Geburtsort, Wohnort, Herkunft). NICHT: allgemeine Orts-, Wetter-, Verkehrs-, Reise- oder Veranstaltungsangaben.
- (DATUM) nur personenbezogene Datumsangaben: Geburtsdaten, Termine oder Zeiträume im Gesundheits- oder Privatkontext (Arzttermin, Krankschreibung, Urlaub, Elternzeit). NICHT: Dokument-, Protokoll-, Liefer-, Rechnungs- oder Veranstaltungsdaten und relative Angaben (heute, gestern, morgen).
- (TELEFON) Telefon- und Faxnummern. (EMAIL) E-Mail-Adressen. (IBAN) Kontonummern. (KARTE) Kreditkartennummern.
- (ID) personenbezogene Kennnummern: Personal-, Kunden-, Vertrags-, Versicherten-, Steuer-, Sozialversicherungs-, Rechnungs-, Bestell-, Schadens-, Handelsregister-, USt-Nummern. NICHT: interne Ticket- oder Vorgangsnummern.
- (CREDENTIAL) Passwörter, Benutzernamen, API-Schlüssel, Tokens, PINs.
- (NETZ) IP-Adressen und interne Server-/Hostnamen. NICHT: öffentliche Websites.
- (KFZ) Kfz-Kennzeichen.
- (GESUNDHEIT) Gesundheitsangaben: Krankheiten, Diagnosen, Behandlungen, Operationen, Medikamente, Allergien, Behinderungen, Schwangerschaft/Mutterschutz, Krankschreibungen.
- (FINANZ) Gehalt, Einkommen, Vermögen, Schulden einer Person.
- (GESCHAEFTSZAHL) vertrauliche Geschäftszahlen: Umsatz, Gewinn, Auftragsvolumen, vertrauliche Preise oder Margen.
- (MERKMAL) weitere persönliche Merkmale, die (in Kombination) identifizieren oder sensibel sind: Alter, Geburtsjahr, Familienstand, Elternzeit, Religion, ethnische Herkunft, politische Meinung, Gewerkschaft, sexuelle Orientierung.

NICHT maskieren: Pronomen, Anreden allein, Rollen- und Funktionsbezeichnungen (z. B. Geschäftsführer, Fahrer, Projektleitung, Kunde), Berufe und Fachgebiete, allgemeine Wörter, Uhrzeiten ohne Personenbezug."""

SYSTEM_DETECT = f"""Du bist ein sorgfältiger Datenschutz-Prüfer (DSGVO). Finde im Text alle Angaben, die maskiert werden müssen, damit keine Person identifiziert werden kann und keine vertraulichen Informationen preisgegeben werden.

{POLICY}

Regeln für die Ausgabe:
- Gib jede Angabe als wörtliche Teilzeichenkette des Textes zurück (exakt, ohne Umformulierung, ohne Ergänzungen).
- Gib Personennamen ohne Anrede/Titel zurück, aber vollständig (Vor- und Nachname, wenn vorhanden).
- Erwähnungen derselben Person in anderer Form (z. B. nur Nachname) jeweils zusätzlich ausgeben.
- Die Hinweisliste stammt von einer automatischen Vorerkennung und kann Fehler und Lücken enthalten: übernimm richtige Hinweise, ignoriere falsche, ergänze Fehlendes.
- Enthält der Text nichts davon, gib eine leere Liste zurück.
Antworte ausschließlich mit JSON nach dem Schema {{"entities": [{{"text": "...", "type": "..."}}]}}."""

SYSTEM_RESIDUAL = f"""Du bist ein Angreifer, der prüft, ob ein bereits pseudonymisierter Text noch Rückschlüsse auf Personen oder vertrauliche Angaben zulässt. Platzhalter in Klammern wie [PERSON_1] oder [ADRESSE_2] sind bereits geschützt – ignoriere sie.

Melde nur Angaben, die laut folgender Richtlinie noch maskiert werden müssten:

{POLICY}

Regeln: wörtliche Teilzeichenketten des Textes, keine Platzhalter, keine Rollenbezeichnungen. Wenn nichts mehr zu maskieren ist, gib eine leere Liste zurück.
Antworte ausschließlich mit JSON nach dem Schema {{"entities": [{{"text": "...", "type": "..."}}]}}."""


class OllamaLLM:
    def __init__(self, host: str, model: str, think: bool = False, temperature: float = 0.0,
                 seed: int = 42, num_ctx: int = 8192, timeout: int = 300):
        self.host = host.rstrip("/")
        if not re.match(r"^https?://", self.host):
            self.host = "http://" + self.host
        self.model = model
        self.think = think
        self.options = {"temperature": temperature, "seed": seed, "num_ctx": num_ctx}
        self.timeout = timeout
        self.calls = 0
        self._send_think = True

    # ------------------------------------------------------------------ HTTP
    def _request(self, path: str, body: dict | None = None, timeout: int | None = None) -> dict:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.host + path, data=data,
                                     headers={"Content-Type": "application/json"},
                                     method="POST" if body is not None else "GET")
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as ex:
            detail = ex.read().decode("utf-8", "replace")[:300]
            raise LLMUnavailable(f"HTTP {ex.code} von {self.host}{path}: {detail}") from ex
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as ex:
            raise LLMUnavailable(f"Ollama unter {self.host} nicht erreichbar: {ex}") from ex

    def check(self) -> None:
        tags = self._request("/api/tags", timeout=10)
        names = {m.get("name") for m in tags.get("models", [])} | {m.get("model") for m in tags.get("models", [])}
        wanted = {self.model, self.model + ":latest"} if ":" not in self.model else {self.model}
        if not names & wanted:
            avail = ", ".join(sorted(n for n in names if n)) or "(keine)"
            raise LLMUnavailable(f"Modell '{self.model}' auf {self.host} nicht gefunden. Verfügbar: {avail}")

    def _chat(self, system: str, user: str, schema: dict | None) -> str:
        body = {"model": self.model, "stream": False, "keep_alive": "15m", "options": self.options,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if schema is not None:
            body["format"] = schema
        if self._send_think:
            body["think"] = self.think
        try:
            res = self._request("/api/chat", body)
        except LLMUnavailable as ex:
            if self._send_think and "think" in str(ex).lower():
                self._send_think = False         # Modell kennt den think-Parameter nicht
                body.pop("think", None)
                res = self._request("/api/chat", body)
            else:
                raise
        self.calls += 1
        return (res.get("message") or {}).get("content", "")

    # --------------------------------------------------------------- Public
    def extract(self, text: str, hints: list[dict] | None = None, residual: bool = False) -> list[dict]:
        if residual:
            system, user = SYSTEM_RESIDUAL, f"Text:\n<<<\n{text}\n>>>"
        else:
            hint_json = json.dumps(hints or [], ensure_ascii=False)
            system = SYSTEM_DETECT
            user = f"Text:\n<<<\n{text}\n>>>\n\nHinweise der Vorerkennung: {hint_json}"
        raw = self._chat(system, user, SCHEMA)
        return _parse_entities(raw)

    def respond(self, system: str, user: str) -> str:
        return self._chat(system, user, None)


def _parse_entities(raw: str) -> list[dict]:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return []
    items = data.get("entities", []) if isinstance(data, dict) else []
    out = []
    for it in items:
        if isinstance(it, dict) and isinstance(it.get("text"), str) and it.get("type") in TYPES:
            out.append({"text": it["text"], "type": it["type"]})
    return out
