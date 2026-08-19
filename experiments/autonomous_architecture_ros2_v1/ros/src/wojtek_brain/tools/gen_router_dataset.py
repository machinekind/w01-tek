#!/usr/bin/env python3
"""Generate the synthetic Polish intent-classification set for the router.

Parameterized tool, not an experiment record: everything enters through
environment variables (declared below).  Points at any OpenAI-compatible
endpoint (vLLM on the GPU box, or a hosted API) and writes JSONL
{"text": ..., "label": ...} ready for train_router.py.  Review the output
by hand before training — a router is only as sane as its dataset.

Usage:
  ROUTER_GEN_URL=http://127.0.0.1:8091 ROUTER_GEN_MODEL=<served-model> \
    python3 gen_router_dataset.py > router_dataset.jsonl
"""

import json
import os
import sys

URL = os.environ.get("ROUTER_GEN_URL") or sys.exit("ROUTER_GEN_URL required")
MODEL = os.environ.get("ROUTER_GEN_MODEL") or sys.exit("ROUTER_GEN_MODEL required")
PER_CLASS = int(os.environ.get("ROUTER_GEN_PER_CLASS", "300"))
TEMPERATURE = float(os.environ.get("ROUTER_GEN_TEMPERATURE", "1.0"))
BATCH = 25  # utterances per request; keeps single responses parseable

CLASSES = {
    "chat": (
        "swobodna rozmowa z robotem-pieskiem: powitania, pytania o samopoczucie, "
        "o to kim jest, co potrafi, ogólna wiedza, żarty, komplementy, small talk"
    ),
    "nav": (
        "polecenia ruchu i szukania: idź/podejdź/znajdź/obróć się/okrąż/wróć, "
        "z celami typu krzesło, kuchnia, piłka, osoba; także wieloetapowe trasy "
        "(np. obejdź stół i stań przy drzwiach)"
    ),
    "visual": (
        "pytania o bieżący widok z kamery: co widzisz, opisz otoczenie, co jest "
        "przed tobą, czy widzisz gdzieś X, rozejrzyj się"
    ),
    "cancel": (
        "przerwanie akcji: stop, stój, przestań, anuluj, zatrzymaj się, wystarczy, "
        "nie idź tam"
    ),
    "system": (
        "polecenia systemowe: zresetuj się, głośniej, ciszej, zmień głos, wyłącz "
        "mikrofon"
    ),
}

PROMPT = (
    "Generujesz dane treningowe dla klasyfikatora intencji robota-pieska "
    "sterowanego głosem po polsku. Wypisz {n} RÓŻNORODNYCH krótkich wypowiedzi "
    "użytkownika należących do klasy: {desc}. Mieszaj rejestry (potoczny, "
    "grzeczny, dziecięcy), długości (1-15 słów) i szyk zdania. Uwzględnij "
    "błędy ASR: brak interpunkcji, małe litery. Wypisz WYŁĄCZNIE wypowiedzi, "
    "jedna na linię, bez numeracji."
)


def generate(label: str, desc: str, n: int) -> list[str]:
    import httpx

    out: list[str] = []
    while len(out) < n:
        r = httpx.post(
            f"{URL.rstrip('/')}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [
                    {"role": "user", "content": PROMPT.format(n=BATCH, desc=desc)}
                ],
                "temperature": TEMPERATURE,
                "max_tokens": 1200,
            },
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
        lines = [ln.strip(" -•\t") for ln in text.splitlines()]
        fresh = [ln for ln in lines if 2 <= len(ln.split()) <= 20 and ln not in out]
        out.extend(fresh)
        print(f"{label}: {len(out)}/{n}", file=sys.stderr)
    return out[:n]


def main():
    for label, desc in CLASSES.items():
        for text in generate(label, desc, PER_CLASS):
            print(json.dumps({"text": text, "label": label}, ensure_ascii=False))


if __name__ == "__main__":
    main()
