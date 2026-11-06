---
name: pseudonymizer
description: Pseudonymisiert personenbezogene und vertrauliche Daten (DSGVO) in deutschen und englischen Texten lokal und führt die Platzhalter in Antworten wieder auf die Originalwerte zurück. Hybrid aus Regex mit Prüfsummen, GLiNER-NER und lokalem LLM (Ollama). Verwenden, wenn Texte, E-Mails, Tickets, Verträge, Protokolle oder Notizen vor der Weiterverarbeitung von Namen, Adressen, Kontaktdaten, IBANs, Kennnummern, Gesundheitsdaten, Passwörtern oder Geschäftszahlen befreit werden sollen ("anonymisieren", "pseudonymisieren", "PII entfernen", "schwärzen", "DSGVO-konform machen"), oder wenn pseudonymisierte Antworten mit [PERSON_1]-artigen Platzhaltern re-identifiziert werden sollen.
---

# Pseudonymizer

Ersetzt sensible Angaben durch typisierte, konsistente Platzhalter (`[PERSON_1]`, `[IBAN_1]` …),
speichert die Zuordnung lokal in einem Tresor und führt Platzhalter in beliebigen Texten –
insbesondere LLM-Antworten – wieder zurück.

## Grundregel: Originaldaten nicht in den eigenen Kontext holen

Der Zweck des Skills ist, dass sensible Inhalte das lokale System nicht verlassen. Deshalb:

1. **Die Originaldatei NICHT selbst lesen** (kein Read/cat/type) – nur das Skript darauf ansetzen.
2. Nur die **pseudonymisierte** Ausgabedatei lesen und damit arbeiten.
3. In eigenen Antworten und Dateien die Platzhalter **unverändert** übernehmen (`[PERSON_1]`, nicht „Person 1“).
4. Nach `restore` die zurückgeführte Datei **nicht** lesen oder ausgeben, sondern dem Nutzer den Pfad nennen –
   außer der Nutzer verlangt ausdrücklich die Anzeige.
5. Den Tresor (`~/.pseudonymizer/vaults/`) niemals lesen, ausgeben oder ins Repository kopieren.

Die Skript-Ausgabe (stdout) enthält nie Originalwerte, nur Pfade, Zähler und Platzhalter.

## Schreiben mit Platzhaltern (für Texte, die später zurückgeführt werden)

`restore` ersetzt **jeden** Platzhalter, der im Tresor steht, auch in Erklärungen und Beispielen.
Außerdem ist unbekannt, welche Sprache, Länge oder Wortart der Originalwert hat. Deshalb gilt:

1. **Platzhalter nie als Beispiel oder Erklärung verwenden.** Falsch: „Platzhalter wie [PERSON_1]
   bleiben erhalten“ (wird zu „Platzhalter wie Harald Schneider …“). Richtig: „Platzhalter in
   eckigen Klammern bleiben erhalten“ – ganz ohne konkreten Schlüssel.
2. **Platzhalter nur dort einsetzen, wo der echte Wert gemeint ist** – nicht umschreiben, nicht
   übersetzen, nicht flektieren (kein „[PERSON_1]s“ statt „von [PERSON_1]“).
3. **Sätze so bauen, dass jeder Wert hineinpasst.** Der Wert kann ein Wort, eine Phrase oder ein
   Satzteil in einer anderen Sprache sein. Platzhalter von Inhaltskategorien (`GESUNDHEIT`,
   `MERKMAL`, `FINANZ`, `GESCHAEFTSZAHL`, `DATUM`) als eigenständige Angabe setzen statt in den
   Satzfluss einzubetten:
   - Falsch: „Catering entsprechend [GESUNDHEIT_4] anpassen.“
   - Richtig: „Catering anpassen (Hinweis: [GESUNDHEIT_4]).“
   - Falsch: „Sie ist [MERKMAL_1] und …“ – Richtig: „Merkmal laut Text: [MERKMAL_1].“
   Personen-, Firmen-, Adress- und Kontaktplatzhalter (`PERSON`, `ORG`, `ADRESSE`, `EMAIL`,
   `TELEFON` …) verhalten sich wie Eigennamen und können normal im Satz stehen.
4. **Lieber weglassen als raten:** Angaben, die für die Aufgabe nicht nötig sind (Passwörter,
   Geburtsdaten, Adressen), nicht in die Ausgabe übernehmen (Datenminimierung).

## Aufruf

Python aus dem Projekt-venv verwenden (Windows: `.venv\Scripts\python.exe`, sonst `.venv/bin/python`).
`SKILL` = `.claude/skills/pseudonymizer`.

```bash
# 1) Pseudonymisieren -> <datei>.pseudo.<ext>, Zuordnung im Tresor
PY SKILL/scripts/pseudo_cli.py pseudonymize eingang.txt --vault projekt-x

# 2) Mit der pseudonymisierten Datei arbeiten, Ergebnis mit Platzhaltern speichern, z. B. antwort.md

# 3) Zurückführen -> <datei>.restored.<ext>
PY SKILL/scripts/pseudo_cli.py restore antwort.md --vault projekt-x --strict
```

Ausgabe von `pseudonymize` (JSON): `output`, `vault_id`, `replacements`, `by_type`, `by_source`,
`llm_used`, `ner_used`, `warnings`. **`warnings` immer prüfen und dem Nutzer melden**, besonders
„LLM nicht verfügbar“ (dann fehlt die Kontextstufe → Gesundheitsdaten/Quasi-Identifikatoren werden schlechter erkannt).

Ausgabe von `restore` (JSON): `mode` (`exakt` = unverändertes eigenes Dokument, byte-genau;
`heuristisch` = sonstiger Text/LLM-Antwort), `restored`, `unresolved` (erfundene/unbekannte Platzhalter
bleiben stehen und werden hier gelistet – dem Nutzer melden).

Wichtige Optionen:

| Option | Wirkung |
|---|---|
| `--vault ID` | Tresor wiederverwenden → gleiche Entität bekommt in mehreren Dateien denselben Platzhalter |
| `--mode redact` | irreversibel `[PERSON]` ohne Nummer, kein Tresor (echte Anonymisierung der Textoberfläche) |
| `--exclude-types GESCHAEFTSZAHL,MERKMAL` | Kategorien nicht maskieren |
| `--require-llm` | Abbruch (Exit 4) statt Weiterarbeiten ohne LLM |
| `--no-llm` / `--no-ner` | Stufen abschalten (schneller, weniger gründlich) |
| `--model`, `--ollama-host` | anderes Ollama-Modell / anderen Host verwenden |
| `--strict` (restore) | Exit 3, wenn unaufgelöste Platzhalter vorkommen |

Weitere Befehle: `vaults list | purge [--older-than TAGE] | delete --id ID`, `check` (prüft NER-Modell und LLM).

Exit-Codes: 0 ok · 2 Eingabe-/Tresorfehler · 3 unaufgelöste Platzhalter (`--strict`) · 4 LLM erforderlich, aber nicht erreichbar.

## Kategorien

`PERSON, ORG, ADRESSE, ORT, DATUM, TELEFON, EMAIL, IBAN, KARTE, ID, GESUNDHEIT, CREDENTIAL, NETZ, KFZ,
FINANZ, GESCHAEFTSZAHL, MERKMAL`. Anreden/Titel bleiben vor dem Platzhalter stehen („Frau Dr. [PERSON_1]“).
Nicht maskiert werden u. a. Rollen, öffentliche Websites, allgemeine Orts-/Datumsangaben, interne Ticketnummern.

## Konfiguration

`SKILL/config.json`: Ollama-Host/Modell (Standard `chat:latest`), GLiNER-Modell und Schwellen,
Residual-Runden, Tresor-Verzeichnis, deaktivierte Kategorien. Host-Priorität:
`--ollama-host` > `PSEUDO_OLLAMA_HOST` > `config.json` > `OLLAMA_HOST` > `http://localhost:11434`.

**Achtung:** Das Erkennungs-LLM sieht die *Originaldaten*. Es muss auf einem vertrauenswürdigen
System laufen (eigener Rechner oder eigenes Netz). Ist ein entfernter Host konfiguriert, den Nutzer
darauf hinweisen.

Tresor verschlüsseln (AES-256-GCM): Umgebungsvariable `PSEUDO_VAULT_PASSPHRASE` setzen – vor
`pseudonymize` *und* `restore`.

## Grenzen (dem Nutzer bei Bedarf mitteilen)

- Keine automatische Erkennung ist vollständig; bei sensiblen Daten Stichprobe durch einen Menschen.
- Pseudonymisierte Daten bleiben nach DSGVO personenbezogen (Art. 4 Nr. 5); erst `--mode redact`
  ohne Tresor nähert sich Anonymisierung – Kontext kann trotzdem re-identifizierend wirken.
- Rückführung funktioniert nur für Platzhalter, die das nachgelagerte Modell wörtlich (oder in
  tolerierter Schreibweise) übernimmt; umschriebene Angaben („der Kunde“) können nicht zurückgeführt werden.

## Einrichtung, Qualitätsprüfung, Methodik

- Einrichtung und Tests: [references/setup.md](references/setup.md)
- Verfahren, Designentscheidungen, Messergebnisse und Quellen: [references/methodik.md](references/methodik.md)
