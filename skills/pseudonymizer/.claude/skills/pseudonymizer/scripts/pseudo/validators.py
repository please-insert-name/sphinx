"""Prüfsummen-Validatoren für strukturierte Kennungen.

Sie erhöhen die Präzision der Regex-Erkennung (vgl. Presidio-Validatoren,
PII-Shield-Länder-Recognizer). Schlägt die Prüfsumme fehl, wird ein Treffer nur
mit passendem Kontextwort (z. B. "IBAN", "Steuer-ID") akzeptiert.
"""
from __future__ import annotations

import re
from collections import Counter


def digits_only(s: str) -> str:
    return re.sub(r"\D", "", s)


def luhn_valid(number: str) -> bool:
    d = digits_only(number)
    if not 12 <= len(d) <= 19:
        return False
    total = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


IBAN_LENGTHS = {
    "AT": 20, "BE": 16, "BG": 22, "CH": 21, "CY": 28, "CZ": 24, "DE": 22, "DK": 18,
    "EE": 20, "ES": 24, "FI": 18, "FR": 27, "GB": 22, "GR": 27, "HR": 21, "HU": 28,
    "IE": 22, "IS": 26, "IT": 27, "LI": 21, "LT": 20, "LU": 20, "LV": 21, "MT": 31,
    "NL": 18, "NO": 15, "PL": 28, "PT": 25, "RO": 24, "SE": 24, "SI": 19, "SK": 24,
}


def iban_valid(s: str) -> bool:
    iban = re.sub(r"\s", "", s).upper()
    if not re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]{10,30}", iban):
        return False
    expected = IBAN_LENGTHS.get(iban[:2])
    if expected and len(iban) != expected:
        return False
    rearranged = iban[4:] + iban[:4]
    number = "".join(str(int(c, 36)) for c in rearranged)
    return int(number) % 97 == 1


def steuer_id_valid(s: str) -> bool:
    """Deutsche steuerliche Identifikationsnummer (11 Ziffern, ISO 7064 MOD 11,10)."""
    d = digits_only(s)
    if len(d) != 11 or d[0] == "0":
        return False
    counts = Counter(d[:10])
    repeated = [c for c in counts.values() if c > 1]
    if len(repeated) != 1 or repeated[0] not in (2, 3):
        return False
    product = 10
    for ch in d[:10]:
        total = (int(ch) + product) % 10
        if total == 0:
            total = 10
        product = (total * 2) % 11
    check = 11 - product
    if check == 10:
        check = 0
    return check == int(d[10])


def svnr_valid(s: str) -> bool:
    """Deutsche Sozialversicherungs-/Rentenversicherungsnummer (12 Zeichen)."""
    s = re.sub(r"\s", "", s).upper()
    m = re.fullmatch(r"(\d{2})(\d{6})([A-Z])(\d{2})(\d)", s)
    if not m:
        return False
    letter = ord(m.group(3)) - 64
    digits = m.group(1) + m.group(2) + f"{letter:02d}" + m.group(4)
    weights = (2, 1, 2, 5, 7, 1, 2, 1, 2, 1, 2, 1)
    total = 0
    for ch, w in zip(digits, weights):
        p = int(ch) * w
        total += p // 10 + p % 10
    return total % 10 == int(m.group(5))


def ipv4_valid(s: str) -> bool:
    parts = s.split(".")
    if len(parts) != 4:
        return False
    for p in parts:
        if not p.isdigit() or int(p) > 255 or (len(p) > 1 and p.startswith("0")):
            return False
    return True
