# Methodik, Designentscheidungen und Messergebnisse

Nummern in [eckigen Klammern] verweisen auf `quellen.txt` im Projektordner.

## Pipeline

| Stufe | Modul | Verfahren | Quelle |
|---|---|---|---|
| 1 Regex | `regex_rules.py` | Muster für E-Mail, Telefon (DE/intl.), IBAN (MOD 97), Kreditkarte (Luhn + Kontext), Steuer-ID (ISO 7064), SV-Nummer (DRV-Prüfziffer), KV-Nr., USt-IdNr., Handelsregister, IPv4/v6, interne Hosts, Passwörter/Benutzernamen/API-Keys, Kfz-Kennzeichen, Geburtsdatum (nur mit Kontext), Gehalt (nur mit Kontext), DE/UK-Adressen, Lexikon für Gesundheits-/Statusbegriffe | [2][15][23][28–31] |
| 2 NER | `ner.py` | GLiNER Zero-Shot mit 14 natürlichsprachlichen Labels | [9][15] |
| 3 LLM | `llm.py` | Ollama mit JSON-Schema, Temperatur 0; Richtlinie im Prompt; NER/Regex-Kandidaten als Hinweise zur Verifikation und Ergänzung | [5][11][12] |
| 4 Fusion | `pipeline.py` | Cluster überlappender Treffer: Regex-Anker > längster Treffer; Reste längerer Modell-Treffer werden nachgezogen. NER-Treffer weicher Typen (ORG, ORT, DATUM …) zählen nur, wenn das LLM sie bestätigt; harte Typen und sichere Personen (≥ 0,85) immer | [2] |
| 5 Verknüpfung | `linking.py` | stabile Platzhalter je Entität; Namensvarianten (Titel/Anrede, nur Nachname, Initiale); gleicher Nachname + anderer Vorname = andere Person; Rechtsformen bei Firmen ignoriert; Telefon normalisiert (+49 ≙ 0) | [2][15] |
| 6 Propagation | `linking.py` | weitere Vorkommen bekannter Werte und „Herr/Frau + Nachname“ ersetzen | [13] |
| 7 Residual | `pipeline.py` | LLM als Angreifer auf dem pseudonymisierten Text; Funde werden nachmaskiert | [5][6] |
| 8 Rückführung | `restore.py` | exakt (Dokument-Hash + Originalschreibweise je Vorkommen) oder heuristisch (tolerante Platzhaltererkennung, Validierung gegen Tresor, Unbekanntes bleibt stehen) | [1][3] |
| 9 Tresor | `vault.py` | JSON oder AES-256-GCM (scrypt), außerhalb des Projekts, TTL/purge | [4][15] |

## Wichtige Designentscheidungen

- **Platzhalter statt Surrogate.** Nutzerwunsch. Außerdem eindeutig erkennbar und damit sicher
  zurückführbar. Surrogate liefern zwar bessere Nutzbarkeit [4], erschweren aber die Rückführung,
  weil das LLM Fake-Namen flektieren oder abkürzen kann.
- **Anrede und Titel bleiben vor dem Platzhalter.** „Frau Dr. [PERSON_1]“ bleibt grammatisch und
  verrät nichts. Im Antwort-Modus wird „Herr [PERSON_1]“ zu „Herr Schneider“, aber nur wenn diese
  Kurzform im Original vorkam (sonst ginge z. B. „seine Frau Petra Ortmann“ verloren).
- **Regex hat bei Überlappung Vorrang.** Grenzen strukturierter Daten sind dort exakt; modellbasierte
  Treffer liefern sonst gern zu viel oder zu wenig.
- **Ein LLM-Ausfall führt nicht zum Abbruch.** Weiterarbeit mit Warnung. `--require-llm` erzwingt den
  Abbruch. Der Grund: Regex + NER allein liefern noch einen brauchbaren Schutz (siehe Messwerte).
- **Persistenter Tresor.** Abweichend von der In-Memory-Empfehlung in [2], weil der Skill die
  Rückführung zeitversetzt braucht. Ausgleich: Ablage außerhalb des Projekts, optional verschlüsselt,
  TTL, und stdout gibt nie Originalwerte aus.
- **Erkennungs-LLM sieht Originaldaten.** Es muss vertrauenswürdig und lokal bzw. im eigenen Netz
  betrieben werden.

## Messergebnisse (06.11.2026)

Ground Truth: `ground_truth.txt` (10 Absätze, 58 Pflicht-, 20 Negativ-, 15 Grenzfall-Einträge) und
Hold-out `tests/holdout.txt` (5 Absätze, 26/10/7). Bewertung je Absatz als eigenes Dokument.

| Konfiguration | Datensatz | Recall | FP auf Negativen | Präzision | Konsistenz | Round-Trip exakt | Antwort-Rückführung |
|---|---|---|---|---|---|---|---|
| Regex allein | example | 0,53 | 0/20 | 1,00 | 1/2 | 10/10 | – |
| Hybrid, chat:latest, erster Lauf | example | 0,98 (57/58) | 0/20 | 1,00 | 6/6 | 8/10* | 67/67 |
| Hybrid, chat:latest, erster Lauf | Hold-out | 1,00 (26/26) | 0/10 | 1,00 | 4/4 | 3/5* | 29/29 |
| **Hybrid, chat:latest, final** | example | **1,00 (58/58)** | **0/20** | **1,00** | **6/6** | **10/10** | **68/68** |
| **Hybrid, chat:latest, final** | Hold-out | **1,00 (26/26)** | **0/10** | **1,00** | **4/4** | **5/5** | **29/29** |

\* vor Einführung der exakten Dokument-Rückführung (Ursache: Namensform-Heuristik).

Der Hold-out-Wert des ersten Laufs wurde gemessen, bevor irgendetwas auf Basis des Hold-out-Sets
geändert wurde. Danach kamen nur die Rückführungslogik und das Gesundheitslexikon hinzu.

Zusätzlich per CLI geprüft: `example.txt` als **ein** Dokument (70 Ersetzungen, keine Pflichtwerte
und keine Ziffernfolgen ≥ 5 im Output) und eine byte-identische exakte Rückführung (LF-Datei).
Verfälschte Platzhalter in Antworten (`<ORG_1>`, `[person_2]`, `\[IBAN_1\]`, `PERSON_3`, `⟦NETZ_1⟧`)
werden aufgelöst, `[PERSON_99]` wird gemeldet (`--strict` → Exit 3). Ein verschlüsselter Tresor
enthält keinen Klartext und lässt sich ohne Passphrase nicht öffnen.

**Nicht abgeschlossen:** Lauf mit einem kleinen lokalen Modell (`gemma4:e2b`) und Round-Trip einer
CRLF-Datei. Beide wurden wegen Speichermangel des Rechners abgebrochen und stehen noch aus.

Beide Datensätze sind klein, synthetisch und von derselben Person (dem Entwickler) erstellt.
Die Werte belegen Funktionsfähigkeit, nicht die Fehlerrate auf realen Daten.

## Bekannte Grenzen und offene Punkte

- Laufzeit: GLiNER auf CPU braucht ca. 30 s zum Laden und ca. 1–2 s je Absatz. Das LLM braucht
  ca. 2–4 s je Absatz und Runde.
- Platzhalter-Erkennung in Antworten: Umschreibungen („die Kundin“) werden nicht zurückgeführt.
  Modelle mit niedriger Echo-Rate [3] benötigen ggf. strengere Prompt-Anweisungen.
- Kontextuelle Grenzfälle (Liefer- oder Rechnungsdatum, große Versicherer/Banken, Projektnamen)
  entscheidet das LLM nach der Richtlinie. Die Ergebnisse können je nach Modell variieren.
- Adressen außerhalb DE/UK und Telefonformate ohne führende 0/+ erkennt nur die NER/LLM-Stufe.
- Lange Dokumente werden absatzweise verarbeitet. Entitäten werden dokumentweit verknüpft, sodass
  gleiche Nachnamen in verschiedenen Kontexten einer Person zugeordnet werden können (Warnung
  „mehrdeutige Namenszuordnung“).
