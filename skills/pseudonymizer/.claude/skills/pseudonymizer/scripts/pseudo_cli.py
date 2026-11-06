#!/usr/bin/env python3
"""Pseudonymisierung mit Rückführung – Kommandozeile.

  pseudonymize  Datei pseudonymisieren (Originalwerte -> [TYP_n]), Zuordnung im Tresor
  restore       Platzhalter in einer Datei (z. B. LLM-Antwort) zurückführen
  vaults        Tresore auflisten / löschen / abgelaufene entfernen
  check         Verfügbarkeit von NER-Modell und LLM prüfen

Datenschutz-Grundsatz: Die Standardausgabe enthält NIE Originalwerte, nur Pfade,
Zähler und Platzhalter. Originalwerte stehen nur in der Eingabedatei, im Tresor und
in der zurückgeführten Ausgabedatei.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo.config import load_config                      # noqa: E402
from pseudo.restore import (placeholder_regex, restore_document, restore_text,  # noqa: E402
                            text_hash)
from pseudo.vault import Vault, list_vaults, new_vault_id, purge_vaults  # noqa: E402

EXIT_OK, EXIT_USAGE, EXIT_UNRESOLVED, EXIT_LLM = 0, 2, 3, 4


def _utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass


def _overrides(args) -> dict:
    o: dict = {"llm": {}, "ner": {}}
    if getattr(args, "ollama_host", None):
        o["llm"]["host"] = args.ollama_host
    if getattr(args, "model", None):
        o["llm"]["model"] = args.model
    if getattr(args, "residual_rounds", None) is not None:
        o["llm"]["residual_rounds"] = args.residual_rounds
    if getattr(args, "exclude_types", None):
        o["types_disabled"] = [t.strip().upper() for t in args.exclude_types.split(",") if t.strip()]
    return o


def _passphrase(cfg: dict) -> str | None:
    return os.environ.get(cfg["vault"].get("passphrase_env", "PSEUDO_VAULT_PASSPHRASE")) or None


def _out_path(inp: Path, out: str | None, tag: str) -> Path:
    if out:
        return Path(out)
    return inp.with_name(f"{inp.stem}.{tag}{inp.suffix or '.txt'}")


def _read(path: Path) -> str:
    # newline="" erhält LF/CRLF des Originals -> byte-genaue Rückführung möglich
    with open(path, encoding="utf-8-sig", newline="") as fh:
        return fh.read()


def _write(path: Path, text: str) -> None:
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write(text)


# --------------------------------------------------------------------------- #
def cmd_pseudonymize(args) -> int:
    from pseudo.linking import Linker
    from pseudo.llm import LLMUnavailable
    from pseudo.pipeline import Pseudonymizer

    cfg = load_config(args.config, _overrides(args))
    inp = Path(args.input)
    if not inp.exists():
        print(f"Fehler: Eingabedatei nicht gefunden: {inp}", file=sys.stderr)
        return EXIT_USAGE
    text = _read(inp)

    vault = None
    if args.mode == "pseudo":
        vault = Vault(cfg["vault"]["dir"], args.vault or new_vault_id(), _passphrase(cfg))
        if vault.exists():
            vault.load()
        elif args.vault and args.must_exist:
            print(f"Fehler: Tresor '{args.vault}' existiert nicht.", file=sys.stderr)
            return EXIT_USAGE
    linker = Linker(vault.data.get("linker") if vault else None)

    warnings = []
    if placeholder_regex().search(text):
        cfg["placeholder_format"] = "⟦{key}⟧"
        warnings.append("Eingabe enthält bereits platzhalterähnliche Zeichenfolgen; verwende ⟦TYP_n⟧.")
        if vault:
            vault.data["collision"] = True
    if vault:
        if vault.data.get("documents"):
            cfg["placeholder_format"] = vault.data.get("placeholder_format", cfg["placeholder_format"])
        vault.data["placeholder_format"] = cfg["placeholder_format"]

    try:
        engine = Pseudonymizer(cfg, use_ner=not args.no_ner, use_llm=not args.no_llm,
                               require_llm=args.require_llm, verbose=args.verbose)
    except LLMUnavailable as ex:
        print(f"Fehler: {ex}", file=sys.stderr)
        return EXIT_LLM
    result = engine.run(text, linker, mode=args.mode)

    out = _out_path(inp, args.output, "pseudo" if args.mode == "pseudo" else "redacted")
    _write(out, result.text)
    if vault:
        vault.data["linker"] = linker.to_dict()
        vault.data["documents"].append({"input": str(inp.resolve()), "output": str(out.resolve()),
                                        "replacements": result.stats["replacements"],
                                        "sha256": text_hash(result.text),
                                        "surfaces": result.surfaces})
        vpath = vault.save()
        if not vault.passphrase:
            warnings.append("Tresor ist unverschlüsselt (Passphrase-Variable "
                            f"{cfg['vault']['passphrase_env']} setzen für AES-256-GCM).")
    summary = {
        "output": str(out.resolve()),
        "vault_id": vault.id if vault else None,
        "vault_path": str(vpath) if vault else None,
        "mode": args.mode,
        "replacements": result.stats["replacements"],
        "entities": result.stats["entities"],
        "by_type": result.stats["by_type"],
        "by_source": result.stats["by_source"],
        "llm_used": result.stats["llm_used"],
        "ner_used": result.stats["ner_used"],
        "residual_added": result.stats.get("residual_added", 0),
        "seconds": result.stats["seconds"],
        "warnings": warnings + result.warnings,
    }
    if args.report:
        Path(args.report).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.print:
        print("\n----- pseudonymisierter Text -----")
        print(result.text)
    return EXIT_OK


def cmd_restore(args) -> int:
    cfg = load_config(args.config)
    inp = Path(args.input)
    if not inp.exists():
        print(f"Fehler: Eingabedatei nicht gefunden: {inp}", file=sys.stderr)
        return EXIT_USAGE
    try:
        vault = Vault(cfg["vault"]["dir"], args.vault, _passphrase(cfg)).load()
    except (FileNotFoundError, PermissionError, ValueError) as ex:
        print(f"Fehler: {ex}", file=sys.stderr)
        return EXIT_USAGE
    text = _read(inp)
    exact = None
    for doc in vault.data.get("documents", []):
        exact = restore_document(text, doc, vault.data.get("placeholder_format", "[{key}]"))
        if exact:
            break
    if exact:
        restored, report = exact
    elif vault.data.get("collision"):
        # Nur das eindeutige Format zurückführen, wörtliche [..]-Folgen unangetastet lassen
        import re
        ents = vault.entities

        def repl(m):
            key = m.group(1)
            return ents[key]["value"] if key in ents else m.group(0)
        restored = re.sub(r"⟦([A-Z]+_\d+)⟧", repl, text)
        left = re.findall(r"⟦[A-Z]+_\d+⟧", restored)
        report = {"mode": "eindeutiges Format ⟦⟧", "placeholders_found": None, "restored": None,
                  "unresolved": left, "keys_used": [], "keys_unused": []}
    else:
        restored, report = restore_text(text, vault.entities)
    out = _out_path(inp, args.output, "restored")
    _write(out, restored)
    summary = {"output": str(out.resolve()), "vault_id": vault.id, **report}
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.print:
        print("\n----- zurückgeführter Text -----")
        print(restored)
    if report["unresolved"] and args.strict:
        return EXIT_UNRESOLVED
    return EXIT_OK


def cmd_vaults(args) -> int:
    cfg = load_config(args.config)
    vdir = cfg["vault"]["dir"]
    if args.action == "list":
        print(json.dumps(list_vaults(vdir), ensure_ascii=False, indent=2))
    elif args.action == "purge":
        days = args.older_than if args.older_than is not None else cfg["vault"].get("ttl_days", 30)
        print(json.dumps({"removed": purge_vaults(vdir, days)}, ensure_ascii=False))
    elif args.action == "delete":
        if not args.id:
            print("Fehler: --id angeben.", file=sys.stderr)
            return EXIT_USAGE
        removed = []
        for suffix in (".json", ".vault"):
            p = Path(os.path.expanduser(vdir)) / f"{args.id}{suffix}"
            if p.exists():
                p.unlink()
                removed.append(str(p))
        print(json.dumps({"removed": removed}, ensure_ascii=False))
    return EXIT_OK


def cmd_check(args) -> int:
    from pseudo.llm import LLMUnavailable, OllamaLLM
    cfg = load_config(args.config, _overrides(args))
    status = {"llm_host": cfg["llm"]["host"], "llm_model": cfg["llm"]["model"]}
    try:
        OllamaLLM(cfg["llm"]["host"], cfg["llm"]["model"]).check()
        status["llm"] = "ok"
    except LLMUnavailable as ex:
        status["llm"] = f"FEHLER: {ex}"
    try:
        from pseudo.ner import NerDetector
        NerDetector(cfg["ner"]["model"])
        status["ner"] = f"ok ({cfg['ner']['model']})"
    except Exception as ex:  # noqa: BLE001
        status["ner"] = f"FEHLER: {type(ex).__name__}: {ex}"
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return EXIT_OK if status["llm"] == "ok" and status["ner"].startswith("ok") else EXIT_LLM


def main(argv=None) -> int:
    _utf8_stdio()
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", help="Pfad zu config.json (Standard: Skill-Ordner)")
    sub = p.add_subparsers(dest="cmd", required=True)

    ps = sub.add_parser("pseudonymize", help="Datei pseudonymisieren")
    ps.add_argument("input")
    ps.add_argument("-o", "--output", help="Ausgabedatei (Standard: <name>.pseudo.<ext>)")
    ps.add_argument("--vault", help="Tresor-ID (vorhandener Tresor wird weiterverwendet -> konsistente Platzhalter über mehrere Dateien)")
    ps.add_argument("--must-exist", action="store_true", help="Fehler, wenn --vault nicht existiert")
    ps.add_argument("--mode", choices=["pseudo", "redact"], default="pseudo",
                    help="pseudo = umkehrbar [TYP_n] (Standard); redact = irreversibel [TYP], kein Tresor")
    ps.add_argument("--no-llm", action="store_true", help="LLM-Stufe abschalten")
    ps.add_argument("--no-ner", action="store_true", help="GLiNER-Stufe abschalten")
    ps.add_argument("--require-llm", action="store_true", help="Abbruch (Exit 4), wenn das LLM nicht erreichbar ist")
    ps.add_argument("--model", help="Ollama-Modell (überschreibt config)")
    ps.add_argument("--ollama-host", help="Ollama-Host, z. B. http://localhost:11434")
    ps.add_argument("--residual-rounds", type=int, help="Anzahl Residual-Prüfrunden (Standard: config)")
    ps.add_argument("--exclude-types", help="Kategorien nicht maskieren, z. B. GESCHAEFTSZAHL,MERKMAL")
    ps.add_argument("--report", help="Zusammenfassung zusätzlich als JSON-Datei speichern")
    ps.add_argument("--print", action="store_true", help="pseudonymisierten Text auch ausgeben")
    ps.add_argument("-v", "--verbose", action="store_true")
    ps.set_defaults(func=cmd_pseudonymize)

    rs = sub.add_parser("restore", help="Platzhalter zurückführen")
    rs.add_argument("input")
    rs.add_argument("--vault", required=True)
    rs.add_argument("-o", "--output", help="Ausgabedatei (Standard: <name>.restored.<ext>)")
    rs.add_argument("--strict", action="store_true", help="Exit 3, wenn unbekannte Platzhalter vorkommen")
    rs.add_argument("--print", action="store_true",
                    help="zurückgeführten Text ausgeben (enthält Originaldaten!)")
    rs.set_defaults(func=cmd_restore)

    vs = sub.add_parser("vaults", help="Tresore verwalten")
    vs.add_argument("action", choices=["list", "purge", "delete"])
    vs.add_argument("--id")
    vs.add_argument("--older-than", type=float, help="Tage (purge)")
    vs.set_defaults(func=cmd_vaults)

    cs = sub.add_parser("check", help="NER-Modell und LLM prüfen")
    cs.add_argument("--model")
    cs.add_argument("--ollama-host")
    cs.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
