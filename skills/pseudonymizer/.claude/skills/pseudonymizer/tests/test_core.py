"""Unit-Tests ohne Modelle (Regex, Validatoren, Verknüpfung, Rückführung, Tresor).

Ausführen:  python tests/test_core.py      (keine pytest-Abhängigkeit nötig)
"""
from __future__ import annotations

import os
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from pseudo.common import trim_span                                    # noqa: E402
from pseudo.config import load_config                                  # noqa: E402
from pseudo.linking import Linker                                      # noqa: E402
from pseudo.pipeline import Pseudonymizer, render                      # noqa: E402
from pseudo.regex_rules import detect_regex                            # noqa: E402
from pseudo.restore import restore_document, restore_text, text_hash   # noqa: E402
from pseudo.validators import iban_valid, luhn_valid, steuer_id_valid, svnr_valid  # noqa: E402
from pseudo.vault import Vault                                         # noqa: E402


def found(text, typ):
    return [text[s.start:s.end] for s in detect_regex(text) if s.type == typ]


def test_validators():
    assert iban_valid("DE89 3704 0044 0532 0130 00")
    assert not iban_valid("DE89 3704 0044 0532 0130 01")
    assert luhn_valid("4111 1111 1111 1111")
    assert not luhn_valid("4111 1111 1111 1112")
    assert steuer_id_valid("86095742719")         # offizielles BZSt-Beispiel
    assert not steuer_id_valid("86095742718")
    assert not steuer_id_valid("01234567890")


def test_regex_structured():
    assert found("IBAN DE89 3704 0044 0532 0130 00 BIC COBADEFFXXX", "IBAN") == ["DE89 3704 0044 0532 0130 00"]
    assert found("Tel. 0177/3344685 oder", "TELEFON") == ["0177/3344685"]
    assert found("Durchwahl 0711 / 998 877-12.", "TELEFON") == ["0711 / 998 877-12"]
    assert found("unter +49 151 23456789 erreichbar", "TELEFON") == ["+49 151 23456789"]
    assert found("am 01.12.2026 um 10:30", "TELEFON") == []
    assert found("Mail: a.b@example.co.uk.", "EMAIL") == ["a.b@example.co.uk"]
    assert "Kastanienallee 7a, 80331 München" in found("in der Kastanienallee 7a, 80331 München, und", "ADRESSE")
    assert "Winkelgasse 12 in Berlin" in found("aus der Winkelgasse 12 in Berlin. Er", "ADRESSE")
    assert found("am Rathausplatz öffnet", "ADRESSE") == []
    assert found("Kennzeichen B-MK 4521 ", "KFZ") == ["B-MK 4521"]
    assert found("das Passwort \"Sommer2024!\" geht", "CREDENTIAL") == ["Sommer2024!"]
    assert found("das neue Passwort schicken?", "CREDENTIAL") == []
    assert sorted(found("Login: fortmann, PW: Kiwi#2025.", "CREDENTIAL")) == ["Kiwi#2025", "fortmann"]
    assert found("Letzter Login von der IP 1.2.3.4", "CREDENTIAL") == []
    assert found("Letzter Login von der IP 87.123.45.201.", "NETZ") == ["87.123.45.201"]
    assert found("Server db-prod-01.intern.firma.de wurde", "NETZ") == ["db-prod-01.intern.firma.de"]
    assert found("(geb. 14.03.1988, Personalnummer 40817)", "DATUM") == ["14.03.1988"]
    assert "40817" in found("(geb. 14.03.1988, Personalnummer 40817)", "ID")
    assert found("Ticket #48213 – Der Kunde", "ID") == []
    assert found("Ticketnummer 48213", "ID") == []
    assert found("seit dem 02.09. wegen einer OP krankgeschrieben ist", "GESUNDHEIT") == ["krankgeschrieben"]
    assert found("Die Krankenkasse zahlt", "GESUNDHEIT") == []


def test_trim_person():
    t = "Sehr geehrte Frau Dr. Lena Fuchs, hiermit"
    s, e = trim_span(t, t.index("Frau"), t.index(","), "PERSON")
    assert t[s:e] == "Lena Fuchs"
    t2 = "seit dem 02.09. wegen"
    assert t2[slice(*trim_span(t2, 9, 15, "DATUM"))] == "02.09."
    t3 = "am 01.12.2026. Danach"
    assert t3[slice(*trim_span(t3, 3, 14, "DATUM"))] == "01.12.2026"


def test_linking():
    lk = Linker()
    a = lk.resolve("PERSON", "Harald Schneider")
    b = lk.resolve("PERSON", "Schneider")
    c = lk.resolve("PERSON", "Tobias Lang")
    d = lk.resolve("PERSON", "Martina Lang")
    e = lk.resolve("ORG", "Brenner & Söhne KG")
    f = lk.resolve("ORG", "Brenner & Söhne")
    g = lk.resolve("TELEFON", "+49 30 5557781")
    h = lk.resolve("TELEFON", "030 5557781")
    assert a == b == "PERSON_1"
    assert c != d
    assert e == f
    assert g == h


def test_restore_tolerant():
    ents = {"PERSON_1": {"type": "PERSON", "value": "Harald Schneider",
                         "variants": ["Harald Schneider", "Schneider"]},
            "PERSON_2": {"type": "PERSON", "value": "Petra Ortmann", "variants": ["Petra Ortmann"]},
            "IBAN_1": {"type": "IBAN", "value": "DE89 3704 0044 0532 0130 00"}}
    out2, _ = restore_text("Seine Frau [PERSON_2] ruft an.", ents)
    assert out2 == "Seine Frau Petra Ortmann ruft an.", out2
    txt = ("Sehr geehrter Herr [PERSON_1], [person_1] / <PERSON_1> / ⟦PERSON_1⟧ / PERSON_1 / "
           "\\[IBAN_1\\] / [IBAN 1] / [PERSON_7] / [Anlage 1]")
    out, rep = restore_text(txt, ents)
    assert out.startswith("Sehr geehrter Herr Schneider, Harald Schneider / Harald Schneider")
    assert "DE89 3704 0044 0532 0130 00 / DE89 3704 0044 0532 0130 00" in out
    assert rep["unresolved"] == ["[PERSON_7]"]
    assert "[Anlage 1]" in out


def test_roundtrip_regex_only():
    cfg = load_config()
    eng = Pseudonymizer(cfg, use_ner=False, use_llm=False)
    text = ("Bitte überweisen Sie auf DE89 3704 0044 0532 0130 00. Rückfragen an sh@firma.de "
            "oder 0177/3344685. Noch einmal: sh@firma.de.")
    lk = Linker()
    res = eng.run(text, lk)
    assert "DE89" not in res.text and "sh@firma.de" not in res.text
    assert res.text.count("[EMAIL_1]") == 2
    back, rep = restore_text(res.text, lk.entities)
    assert back == text, back
    assert render(text, [], "[{key}]") == text
    # exakte Dokument-Rückführung
    doc = {"sha256": text_hash(res.text), "surfaces": res.surfaces}
    exact, rep2 = restore_document(res.text, doc, "[{key}]")
    assert exact == text and rep2["mode"] == "exakt"
    assert restore_document(res.text + " geändert", doc, "[{key}]") is None


def test_vault_encryption():
    with tempfile.TemporaryDirectory() as d:
        v = Vault(d, "t1", "geheim")
        v.data["linker"] = {"entities": {"PERSON_1": {"type": "PERSON", "value": "X Y"}}}
        p = v.save()
        assert p.suffix == ".vault" and b"X Y" not in p.read_bytes()
        assert Vault(d, "t1", "geheim").load().entities["PERSON_1"]["value"] == "X Y"
        try:
            Vault(d, "t1", "falsch").load()
            raise AssertionError("falsche Passphrase akzeptiert")
        except Exception as ex:   # InvalidTag
            assert "AssertionError" not in type(ex).__name__


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"ok    {t.__name__}")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL  {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} Tests bestanden")
    sys.exit(1 if failed else 0)
