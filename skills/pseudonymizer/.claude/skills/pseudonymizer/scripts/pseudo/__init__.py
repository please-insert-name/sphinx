"""pseudo – hybride Pseudonymisierung (Regex + GLiNER + lokales LLM) mit Rückführung.

Module:
    common      Datentypen, Textsegmentierung, Hilfsfunktionen
    validators  Prüfsummen (IBAN, Luhn, Steuer-ID, SV-Nummer, IPv4)
    regex_rules Regelbasierte Erkennung strukturierter Daten
    ner         GLiNER-Erkennung (Zero-Shot-NER)
    llm         Kontextuelle Erkennung über ein lokales Ollama-Modell
    linking     Entitätsverknüpfung (konsistente Platzhalter) und Propagation
    pipeline    Zusammenführung aller Stufen
    restore     Rückführung von Platzhaltern in (LLM-)Antworten
    vault       Speicherung der Zuordnungstabelle (optional AES-256-GCM)
"""

__version__ = "1.0.0"
