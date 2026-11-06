#!/usr/bin/env python3
"""Qualitätsprüfung des Pseudonymisierers gegen eine Ground-Truth-Datei.

Nur für synthetische Testdaten gedacht – die Ausgabe enthält Originalwerte!

Ground-Truth-Format (siehe ground_truth.txt):
    --- <n> ----          Abschnitt = n-ter Absatz der Textdatei
    [KAT] "Text"          Pflicht-Maskierung
    [KAT] "Text" (!)      Grenzfall (wird protokolliert, nicht bewertet)
    [KAT] "Text" (-)      darf NICHT maskiert werden
    (-) "a", "b"          Negativ-Liste
    Hinweise "(gleiche Entität ...)" / "(... andere Person ...)" erzeugen Konsistenzprüfungen.

Metriken (angelehnt an TAB, Pilán et al. 2022): Recall auf Pflichtangaben
(Token-Abdeckung >= 75 % je Vorkommen), False Positives auf Negativ-Checks,
Präzision (Anteil Maskierungen mit Bezug zu Pflicht/Grenzfall), Konsistenz,
Round-Trip-Treue, optional Rückführung in echten LLM-Antworten.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from pseudo.common import find_all, paragraph_spans          # noqa: E402
from pseudo.config import load_config                        # noqa: E402
from pseudo.linking import Linker                            # noqa: E402
from pseudo.pipeline import Pseudonymizer                    # noqa: E402
from pseudo.restore import (placeholder_regex, restore_document,  # noqa: E402
                            restore_text, text_hash)

QUOTE = re.compile(r'"([^"]+)"')
STOP = {
    "herr", "herrn", "frau", "dr", "prof", "mr", "mrs", "ms", "in", "im", "am", "an", "der", "die", "das",
    "den", "dem", "des", "ein", "eine", "einer", "einen", "wegen", "um", "bei", "von", "vom", "mit",
    "und", "pro", "seit", "the", "of", "to", "is", "at", "for", "and", "a", "uhr", "nach", "auf", "per",
    "ab", "on",
}
COVER_MIN = 0.75


# --------------------------------------------------------------------------- #
def parse_truth(path: Path) -> dict[int, list[dict]]:
    sections: dict[int, list[dict]] = {}
    cur, last, in_note, in_neg = None, [], False, False
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line, s = raw.rstrip(), raw.strip()
        m = re.match(r"^---\s*(\d+)\s*-", line)
        if m:
            cur = int(m.group(1))
            sections[cur] = []
            last, in_note, in_neg = [], False, False
            continue
        if cur is None:
            continue
        if line.startswith("====="):
            cur = None
            continue
        if not s:
            in_note = in_neg = False
            continue

        def add(texts, cat, marker, note=""):
            items = [{"cat": cat, "text": t, "marker": marker, "note": note} for t in texts]
            sections[cur].extend(items)
            return items

        if s.startswith("Anmerkung:"):
            in_note, in_neg = True, False
            if "(-)" in s:
                add(QUOTE.findall(s), None, "neg")
            continue
        if line.startswith("["):
            in_note = in_neg = False
            cat = re.match(r"\[([^\]]+)\]", line).group(1)
            quotes = QUOTE.findall(line)
            tail = line[line.rfind('"') + 1:] if quotes else ""
            marker = "neg" if "(-)" in tail else "border" if "(!)" in tail else "must"
            last = add(quotes, cat, marker, tail.strip())
            continue
        if s.startswith("(-)"):
            in_neg, in_note = True, False
            add(QUOTE.findall(s), None, "neg")
            continue
        if in_note:
            if "(-)" in s:
                add(QUOTE.findall(s), None, "neg")
            continue
        if in_neg:
            add(QUOTE.findall(s), None, "neg")
            continue
        for it in last:   # Fortsetzungszeile eines Eintrags
            if s.startswith("(-)"):
                it["marker"] = "neg"
            elif s.startswith("(!)") and it["marker"] == "must":
                it["marker"] = "border"
            it["note"] = (it["note"] + " " + s).strip()
    return sections


def _tokens(sub: str, base: int, drop_stop: bool):
    toks = [(m.start() + base, m.end() + base, m.group()) for m in re.finditer(r"\w+", sub)]
    if drop_stop:
        kept = [t for t in toks if t[2].casefold() not in STOP]
        return kept or toks
    return toks


def _coverage(occ, masked, drop_stop=True) -> float:
    a, b = occ
    toks = _tokens(masked["text"][a:b], a, drop_stop)
    total = sum(e - s for s, e, _ in toks)
    if total == 0:
        return 0.0
    cov = sum(sum(1 for i in range(s, e) if masked["mask"][i]) for s, e, _ in toks)
    return cov / total


def _key_at(occ, spans):
    a, b = occ
    best, best_ov = None, 0
    for s in spans:
        ov = min(b, s.end) - max(a, s.start)
        if ov > best_ov:
            best, best_ov = s.key, ov
    return best


# --------------------------------------------------------------------------- #
def evaluate(text_path: Path, truth_path: Path, engine: Pseudonymizer, cfg: dict,
             response_test: bool, verbose: bool, show_output: bool) -> dict:
    text = text_path.read_text(encoding="utf-8-sig")
    paras = [text[a:b] for a, b in paragraph_spans(text)]
    truth = parse_truth(truth_path)
    agg = {"must": 0, "hit": 0, "partial": 0, "miss": 0, "neg": 0, "fp": 0, "border": 0,
           "border_masked": 0, "spans": 0, "spans_relevant": 0, "spans_neg": 0, "spans_extra": 0,
           "consistency_ok": 0, "consistency_fail": 0, "roundtrip_ok": 0, "roundtrip_heur_ok": 0,
           "paragraphs": 0,
           "gt_not_found": 0, "resp_placeholders": 0, "resp_restored": 0, "resp_unresolved": 0,
           "resp_leftover": 0}
    details = []
    ph_rx = placeholder_regex()

    for idx, para in enumerate(paras, 1):
        items = truth.get(idx, [])
        linker = Linker()
        t = time.time()
        res = engine.run(para, linker)
        dur = time.time() - t
        mask = [False] * len(para)
        for s in res.spans:
            for i in range(s.start, s.end):
                mask[i] = True
        masked = {"text": para, "mask": mask}
        agg["paragraphs"] += 1
        rep = {"paragraph": idx, "seconds": round(dur, 1), "items": [], "extra": [], "consistency": [],
               "warnings": res.warnings}

        all_occ = []
        prev_by_cat: dict[str, dict] = {}
        for it in items:
            occs = find_all(para, it["text"])
            it_rep = {**it, "occurrences": len(occs)}
            if not occs:
                agg["gt_not_found"] += 1
                it_rep["status"] = "nicht im Text"
                rep["items"].append(it_rep)
                continue
            all_occ.extend(occs)
            covs = [_coverage(o, masked, it["marker"] != "neg") for o in occs]
            keys = [_key_at(o, res.spans) for o in occs]
            it_rep["coverage"] = [round(c, 2) for c in covs]
            it_rep["keys"] = keys
            if it["marker"] == "must":
                agg["must"] += 1
                if all(c >= COVER_MIN for c in covs):
                    status = "hit"
                elif any(c > 0 for c in covs):
                    status = "partial"
                else:
                    status = "miss"
                agg[status] += 1
            elif it["marker"] == "neg":
                agg["neg"] += 1
                status = "fp" if any(_coverage(o, masked, False) > 0 for o in occs) else "ok"
                if status == "fp":
                    agg["fp"] += 1
            else:
                agg["border"] += 1
                status = "maskiert" if any(c >= COVER_MIN for c in covs) else (
                    "teilweise" if any(c > 0 for c in covs) else "nicht maskiert")
                if status == "maskiert":
                    agg["border_masked"] += 1
            it_rep["status"] = status

            # Konsistenz
            if it["marker"] != "neg":
                if len(occs) > 1:
                    ok = len(set(keys)) == 1 and keys[0] is not None
                    rep["consistency"].append((f"alle Vorkommen von '{it['text']}' gleich", ok))
                note = it["note"].lower()
                prev = prev_by_cat.get(it["cat"])
                if prev and "gleiche entität" in note:
                    ok = keys[0] is not None and keys[0] == prev["keys"][0]
                    rep["consistency"].append((f"'{it['text']}' == '{prev['text']}'", ok))
                if prev and "andere person" in note:
                    ok = keys[0] is not None and keys[0] != prev["keys"][0]
                    rep["consistency"].append((f"'{it['text']}' != '{prev['text']}'", ok))
                if keys[0] is not None:
                    prev_by_cat[it["cat"]] = {"text": it["text"], "keys": keys}
            rep["items"].append(it_rep)

        for name, ok in rep["consistency"]:
            agg["consistency_ok" if ok else "consistency_fail"] += 1

        # Präzision der Maskierungen
        must_border = [o for it in items if it["marker"] != "neg" for o in find_all(para, it["text"])]
        negs = [o for it in items if it["marker"] == "neg" for o in find_all(para, it["text"])]
        for s in res.spans:
            agg["spans"] += 1
            if any(s.start < b and a < s.end for a, b in must_border):
                agg["spans_relevant"] += 1
            elif any(s.start < b and a < s.end for a, b in negs):
                agg["spans_neg"] += 1
            else:
                agg["spans_extra"] += 1
                rep["extra"].append(f"[{s.type}/{s.source}] '{para[s.start:s.end]}'")

        # Round-Trip: exakt (Dokument-Modus) und heuristisch (Modus für LLM-Antworten)
        doc = {"sha256": text_hash(res.text), "surfaces": res.surfaces}
        exact = restore_document(res.text, doc, cfg["placeholder_format"])
        rt_ok = exact is not None and exact[0] == para
        agg["roundtrip_ok"] += int(rt_ok)
        rep["roundtrip"] = rt_ok
        heur, _ = restore_text(res.text, linker.entities)
        agg["roundtrip_heur_ok"] += int(heur == para)
        if heur != para:
            rep["roundtrip_heur_diff"] = heur

        # Rückführung in einer echten LLM-Antwort
        if response_test and engine.llm:
            system = ("Du bist Sachbearbeiter in einem Unternehmen. Der Text enthält Platzhalter wie "
                      "[PERSON_1] oder [IBAN_1] anstelle geschützter Angaben. Verwende diese Platzhalter "
                      "in deiner Antwort exakt unverändert, wenn du dich auf die Angaben beziehst.")
            user = ("Schreibe eine kurze Antwort oder Notiz (höchstens 4 Sätze) zu folgender Nachricht "
                    "und nenne dabei die wichtigsten beteiligten Personen und Daten:\n\n" + res.text)
            answer = engine.llm.respond(system, user)
            back, arep = restore_text(answer, linker.entities)
            leftover = len(ph_rx.findall(back))
            agg["resp_placeholders"] += arep["placeholders_found"]
            agg["resp_restored"] += arep["restored"]
            agg["resp_unresolved"] += len(arep["unresolved"])
            agg["resp_leftover"] += leftover
            rep["response"] = {"answer": answer, "restored": back, "unresolved": arep["unresolved"]}

        rep["output"] = res.text
        details.append(rep)

    agg["recall"] = round(agg["hit"] / agg["must"], 3) if agg["must"] else None
    agg["recall_incl_partial"] = round((agg["hit"] + agg["partial"]) / agg["must"], 3) if agg["must"] else None
    agg["precision"] = round(agg["spans_relevant"] / agg["spans"], 3) if agg["spans"] else None
    agg["fp_rate_neg"] = round(agg["fp"] / agg["neg"], 3) if agg["neg"] else None
    _print(agg, details, verbose, show_output, response_test and engine.llm is not None)
    return {"summary": agg, "details": details}


def _print(agg, details, verbose, show_output, resp):
    p = print
    for d in details:
        problems = [i for i in d["items"] if i["status"] in ("miss", "partial", "fp")]
        cons_fail = [c for c in d["consistency"] if not c[1]]
        flag = "OK " if not problems and not cons_fail and d["roundtrip"] else "!! "
        p(f"\n{flag}Absatz {d['paragraph']}  ({d['seconds']}s)")
        for i in d["items"]:
            if verbose or i["status"] in ("miss", "partial", "fp", "nicht im Text"):
                p(f"   {i['status']:<15} [{i['marker']:<6}] {i['cat'] or '-':<12} '{i['text']}' "
                  f"cov={i.get('coverage')} keys={i.get('keys')}")
        for name, ok in d["consistency"]:
            if verbose or not ok:
                p(f"   Konsistenz {'ok  ' if ok else 'FEHLER'}: {name}")
        for e in d["extra"]:
            p(f"   zusätzlich maskiert: {e}")
        if not d["roundtrip"]:
            p("   Round-Trip FEHLER")
        for w in d["warnings"]:
            if "LLM nicht" in w or "NER nicht" in w or "mehrdeutig" in w:
                p(f"   Warnung: {w}")
        if show_output:
            p("   --> " + d["output"])
        if resp and "response" in d and (verbose or d["response"]["unresolved"]):
            p("   LLM-Antwort: " + d["response"]["answer"].replace("\n", " ")[:400])
            p("   zurückgeführt: " + d["response"]["restored"].replace("\n", " ")[:400])
            if d["response"]["unresolved"]:
                p(f"   unaufgelöst: {d['response']['unresolved']}")
    a = agg
    p("\n" + "=" * 70)
    p(f"Pflichtangaben:   {a['hit']}/{a['must']} vollständig, {a['partial']} teilweise, {a['miss']} verfehlt"
      f"  -> Recall {a['recall']} (inkl. teilweise {a['recall_incl_partial']})")
    p(f"Negativ-Checks:   {a['fp']}/{a['neg']} fälschlich maskiert")
    p(f"Grenzfälle:       {a['border_masked']}/{a['border']} maskiert (nicht bewertet)")
    p(f"Maskierungen:     {a['spans']} gesamt, {a['spans_relevant']} mit GT-Bezug, {a['spans_neg']} auf Negativen, "
      f"{a['spans_extra']} ohne GT-Bezug -> Präzision {a['precision']}")
    p(f"Konsistenz:       {a['consistency_ok']} ok, {a['consistency_fail']} Fehler")
    p(f"Round-Trip:       {a['roundtrip_ok']}/{a['paragraphs']} exakt (Dokument-Modus), "
      f"{a['roundtrip_heur_ok']}/{a['paragraphs']} identisch im Antwort-Modus (Namensform-Heuristik)")
    if resp:
        p(f"LLM-Antworten:    {a['resp_restored']}/{a['resp_placeholders']} Platzhalter zurückgeführt, "
          f"{a['resp_unresolved']} unaufgelöst, {a['resp_leftover']} Platzhalter übrig")
    if a["gt_not_found"]:
        p(f"Hinweis:          {a['gt_not_found']} GT-Einträge nicht wörtlich im Text gefunden")


def main(argv=None) -> int:
    for st in (sys.stdout, sys.stderr):
        try:
            st.reconfigure(encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--text", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--config")
    ap.add_argument("--no-llm", action="store_true")
    ap.add_argument("--no-ner", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--ollama-host")
    ap.add_argument("--residual-rounds", type=int)
    ap.add_argument("--response-test", action="store_true", help="Rückführung in echten LLM-Antworten testen")
    ap.add_argument("--json", help="Ergebnisse als JSON speichern")
    ap.add_argument("--show-output", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)
    over = {"llm": {}}
    if args.model:
        over["llm"]["model"] = args.model
    if args.ollama_host:
        over["llm"]["host"] = args.ollama_host
    if args.residual_rounds is not None:
        over["llm"]["residual_rounds"] = args.residual_rounds
    cfg = load_config(args.config, over)
    engine = Pseudonymizer(cfg, use_ner=not args.no_ner, use_llm=not args.no_llm, verbose=args.verbose)
    for w in engine.init_warnings:
        print("Warnung:", w)
    print(f"Stufen: Regex=ja, NER={'ja' if engine.ner else 'nein'}, "
          f"LLM={'ja (' + cfg['llm']['model'] + ' @ ' + cfg['llm']['host'] + ')' if engine.llm else 'nein'}")
    result = evaluate(Path(args.text), Path(args.truth), engine, cfg, args.response_test,
                      args.verbose, args.show_output)
    if args.json:
        Path(args.json).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
