# Einrichtung und Qualitätsprüfung

## Einrichtung

```bash
# im Projektordner
python -m venv --system-site-packages .venv      # nutzt vorhandenes torch mit
.venv/Scripts/python.exe -m pip install -r .claude/skills/pseudonymizer/requirements.txt
.venv/Scripts/python.exe .claude/skills/pseudonymizer/scripts/pseudo_cli.py check
```

`check` lädt das GLiNER-Modell (beim ersten Mal Download, ca. 1,1 GB, danach offline aus dem
Hugging-Face-Cache) und prüft, ob das konfigurierte Ollama-Modell erreichbar ist.

Bekannte Fallstricke:
- `transformers` 5.x: Tokenizer-Konvertierung von mDeBERTa schlägt fehl → im venv `transformers<5`.
- Mit `--system-site-packages` kann ein älteres System-`protobuf` gewinnen →
  `pip install --ignore-installed "protobuf>=7.36.1"` im venv.
- PowerShell-Profil und Git-Bash können unterschiedliche `OLLAMA_HOST`-Werte haben; deshalb steht der
  Host explizit in `config.json`.

## Tests

```bash
# Unit-Tests ohne Modelle (Validatoren, Regex, Verknüpfung, Rückführung, Tresor)
PY tests/test_core.py

# Qualitätsprüfung gegen Ground Truth (gibt Originalwerte aus – nur synthetische Daten!)
PY scripts/evaluate.py --text ../../../example.txt --truth ../../../ground_truth.txt --response-test
PY scripts/evaluate.py --text tests/holdout.txt --truth tests/holdout_truth.txt --response-test
```

`evaluate.py` misst je Absatz (als eigenes Dokument):
- **Recall** auf Pflichtangaben: ein Eintrag zählt als getroffen, wenn je Vorkommen ≥ 75 % der
  inhaltstragenden Zeichen maskiert sind (Anreden/Funktionswörter ausgenommen).
- **False Positives** auf Negativ-Checks `(-)` und **Präzision** (Anteil der Maskierungen mit Bezug zu
  Pflicht-/Grenzfall-Einträgen; Maskierungen ohne GT-Bezug werden einzeln gelistet).
- **Konsistenz** (gleiche Entität → gleicher Platzhalter, verschiedene Personen → verschiedene).
- **Round-Trip** exakt (Dokument-Modus) und im Antwort-Modus.
- `--response-test`: lässt das LLM auf den pseudonymisierten Text antworten und prüft, ob alle
  Platzhalter der Antwort zurückgeführt werden.

`tests/holdout.txt` ist ein Kontrollsatz, auf den der Skill nicht hin optimiert wurde.
