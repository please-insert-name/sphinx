"""Tresor für die Zuordnung Platzhalter -> Originalwert.

Pseudonymisierte Daten bleiben nach DSGVO personenbezogen (Art. 4 Nr. 5); der Tresor
ist der Schlüssel zur Re-Identifikation und muss geschützt werden:
* Ablage außerhalb des Projekts (Standard: ~/.pseudonymizer/vaults), nicht im Git.
* Optional AES-256-GCM-Verschlüsselung mit scrypt-abgeleitetem Schlüssel, wenn die
  Umgebungsvariable PSEUDO_VAULT_PASSPHRASE (konfigurierbar) gesetzt ist
  (vgl. PII Shield, SurrogateShield arXiv:2606.29567).
* Ablaufdatum (TTL) und purge-Befehl.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import secrets
from pathlib import Path

MAGIC = b"PSV1"


def _now() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def new_vault_id() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(3)


def _derive(passphrase: str, salt: bytes) -> bytes:
    return hashlib.scrypt(passphrase.encode("utf-8"), salt=salt, n=2 ** 15, r=8, p=1,
                          maxmem=64 * 1024 * 1024, dklen=32)


def _encrypt(data: bytes, passphrase: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    salt, nonce = os.urandom(16), os.urandom(12)
    return MAGIC + salt + nonce + AESGCM(_derive(passphrase, salt)).encrypt(nonce, data, MAGIC)


def _decrypt(blob: bytes, passphrase: str) -> bytes:
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM
    if not blob.startswith(MAGIC):
        raise ValueError("Keine gültige Tresor-Datei")
    salt, nonce, ct = blob[4:20], blob[20:32], blob[32:]
    return AESGCM(_derive(passphrase, salt)).decrypt(nonce, ct, MAGIC)


class Vault:
    def __init__(self, vault_dir: str, vault_id: str, passphrase: str | None = None):
        self.dir = Path(os.path.expanduser(vault_dir))
        self.id = vault_id
        self.passphrase = passphrase or None
        self.data: dict = {"version": 1, "id": vault_id, "created": _now(), "updated": _now(),
                           "placeholder_format": "[{key}]", "documents": [], "linker": {}}

    @property
    def path(self) -> Path:
        return self.dir / (f"{self.id}.vault" if self.passphrase else f"{self.id}.json")

    def exists(self) -> bool:
        return (self.dir / f"{self.id}.vault").exists() or (self.dir / f"{self.id}.json").exists()

    def load(self) -> "Vault":
        enc, plain = self.dir / f"{self.id}.vault", self.dir / f"{self.id}.json"
        if enc.exists():
            if not self.passphrase:
                raise PermissionError(f"Tresor {self.id} ist verschlüsselt – Passphrase-Variable setzen.")
            self.data = json.loads(_decrypt(enc.read_bytes(), self.passphrase).decode("utf-8"))
        elif plain.exists():
            self.data = json.loads(plain.read_text(encoding="utf-8"))
            self.passphrase = None          # unverschlüsselt bleibt unverschlüsselt
        else:
            raise FileNotFoundError(f"Tresor '{self.id}' nicht gefunden in {self.dir}")
        return self

    def save(self) -> Path:
        self.dir.mkdir(parents=True, exist_ok=True)
        self.data["updated"] = _now()
        raw = json.dumps(self.data, ensure_ascii=False, indent=1).encode("utf-8")
        path = self.path
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(_encrypt(raw, self.passphrase) if self.passphrase else raw)
        os.replace(tmp, path)
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
        return path

    @property
    def entities(self) -> dict:
        return self.data.get("linker", {}).get("entities", {})


def list_vaults(vault_dir: str) -> list[dict]:
    d = Path(os.path.expanduser(vault_dir))
    if not d.exists():
        return []
    out = []
    for p in sorted(d.iterdir()):
        if p.suffix in (".json", ".vault"):
            st = p.stat()
            out.append({"id": p.stem, "encrypted": p.suffix == ".vault",
                        "modified": dt.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
                        "bytes": st.st_size, "path": str(p)})
    return out


def purge_vaults(vault_dir: str, older_than_days: float) -> list[str]:
    cutoff = dt.datetime.now().timestamp() - older_than_days * 86400
    removed = []
    for v in list_vaults(vault_dir):
        p = Path(v["path"])
        if p.stat().st_mtime < cutoff:
            p.unlink()
            removed.append(v["id"])
    return removed
