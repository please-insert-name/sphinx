"""Zero-Shot-NER mit GLiNER (Zaratiana et al., NAACL 2024, arXiv:2311.08526).

Standardmodell: urchade/gliner_multi_pii-v1 (mehrsprachig inkl. Deutsch, PII-feinjustiert).
Die Labels werden in natürlicher Sprache übergeben und auf die kanonischen
Kategorien abgebildet.
"""
from __future__ import annotations

import contextlib
import logging
import os
import sys
import warnings

from .common import Span, chunk_spans

DEFAULT_LABELS = {
    "person": "PERSON",
    "organization": "ORG",
    "street address": "ADRESSE",
    "phone number": "TELEFON",
    "email address": "EMAIL",
    "date of birth": "DATUM",
    "iban": "IBAN",
    "credit card number": "KARTE",
    "medical condition": "GESUNDHEIT",
    "username": "CREDENTIAL",
    "password": "CREDENTIAL",
    "ip address": "NETZ",
    "license plate number": "KFZ",
    "location": "ORT",
}


class NerDetector:
    def __init__(self, model_name: str, labels: dict | None = None, threshold: float = 0.5,
                 max_chunk_chars: int = 1000):
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        warnings.filterwarnings("ignore")
        logging.getLogger("transformers").setLevel(logging.ERROR)
        with contextlib.redirect_stdout(sys.stderr):
            from gliner import GLiNER
            try:   # bevorzugt offline aus dem lokalen Cache
                self.model = GLiNER.from_pretrained(model_name, local_files_only=True)
            except Exception:
                self.model = GLiNER.from_pretrained(model_name)
        self.labels = labels or DEFAULT_LABELS
        self.threshold = threshold
        self.max_chunk_chars = max_chunk_chars
        self.model_name = model_name

    def detect(self, text: str) -> list[Span]:
        out: list[Span] = []
        label_list = list(self.labels)
        for cs, ce in chunk_spans(text, self.max_chunk_chars):
            chunk = text[cs:ce]
            with contextlib.redirect_stdout(sys.stderr):
                ents = self.model.predict_entities(chunk, label_list, threshold=self.threshold)
            for en in ents:
                typ = self.labels.get(en["label"])
                if typ:
                    out.append(Span(cs + en["start"], cs + en["end"], typ, "ner",
                                    float(en["score"]), rule="gliner:" + en["label"]))
        return out
