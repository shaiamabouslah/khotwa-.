"""Retrieval benchmark of what the web app actually does (generate.retrieve: dense score floor,
CNSS role ordering, no document-language filter), per question language, for each retrieval
mode. Written as a DVC metrics file so runs can be compared with `dvc metrics diff`.

  python scripts/benchmark_retrieval.py
  python scripts/benchmark_retrieval.py --modes dense

Positives (data/rag/questions.jsonl): each question has the document that answers it, plus
optional "also_ok" documents (the same service published in the other language). hit@1/3/5 and
MRR, with a 95% bootstrap interval on the overall scores: with ~100 questions one question moves
a score by about one point, so differences inside the interval are noise.
Robustness (data/rag/hard_questions.jsonl): typos, keyword-style and Derja variants, reported apart.
Negatives (data/rag/negative_questions.jsonl): share stopped by the score floor before any LLM
call -- the rest is left to the generation prompt.

Every gold document must exist in data/rag/chunks.jsonl: a label pointing to a removed or renamed
document would silently count as a retrieval miss, so the run stops instead.
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import generate  # noqa: E402
from retriever import Retriever  # noqa: E402

MODES = {"dense": {"hybrid": False}, "hybrid": {"hybrid": True}}
BOOTSTRAP_SAMPLES = 2000


def load(path: str) -> list[dict]:
    with open(ROOT / path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_labels(retriever: Retriever, *question_sets: list[dict]) -> None:
    known = {c["doc_id"] for c in retriever.chunks}
    bad = [(q["q"], d) for qs in question_sets for q in qs
           for d in [q["doc_id"], *q.get("also_ok", [])] if d not in known]
    if bad:
        lines = "\n".join(f"  {d}  <-  {q[:70]}" for q, d in bad)
        sys.exit(f"{len(bad)} benchmark label(s) point to documents that are not in chunks.jsonl:\n{lines}")


def ranked_docs(retriever: Retriever, question: str, hybrid: bool) -> list[str]:
    docs = []
    for h in generate.retrieve(retriever, question, k=5, rerank=False, hybrid=hybrid):
        if h.doc_id not in docs:
            docs.append(h.doc_id)
    return docs


def position(q: dict, docs: list[str]) -> int:
    gold = {q["doc_id"], *q.get("also_ok", [])}
    return next((i for i, d in enumerate(docs, 1) if d in gold), 0)


def _rates(positions: list[int]) -> dict:
    n = len(positions)
    out = {f"hit@{k}": sum(0 < p <= k for p in positions) / n for k in (1, 3, 5)}
    out["mrr"] = sum(1 / p for p in positions if p) / n
    return out


def score(rows: list[dict], with_ci: bool = False) -> dict:
    positions = [r["pos"] for r in rows]
    out = {"n": len(rows), **{k: round(v, 4) for k, v in _rates(positions).items()}}
    if with_ci:
        rng = random.Random(0)
        samples = [_rates(rng.choices(positions, k=len(positions))) for _ in range(BOOTSTRAP_SAMPLES)]
        for metric in ("hit@1", "hit@5", "mrr"):
            vals = sorted(s[metric] for s in samples)
            out[f"{metric}_ci95"] = [round(vals[int(0.025 * len(vals))], 4), round(vals[int(0.975 * len(vals)) - 1], 4)]
    return out


def evaluate(retriever: Retriever, questions: list[dict], hybrid: bool) -> list[dict]:
    rows = []
    for q in questions:
        docs = ranked_docs(retriever, q["q"], hybrid)
        rows.append({"q": q["q"], "lang": q.get("lang", "?"), "pos": position(q, docs), "got": docs[:3]})
    return rows


def run_mode(retriever: Retriever, positives, hard, negatives, hybrid: bool) -> tuple[dict, list]:
    rows = evaluate(retriever, positives, hybrid)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)
    blocked = sum(1 for q in negatives if not generate.retrieve(retriever, q["q"], k=5, rerank=False, hybrid=hybrid))
    result = {"all": score(rows, with_ci=True), **{lang: score(rs) for lang, rs in sorted(by_lang.items())}}
    if hard:
        result["robustness"] = score(evaluate(retriever, hard, hybrid))
    result["negatives_blocked_by_floor"] = round(blocked / len(negatives), 4) if negatives else None
    return result, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="data/rag/questions.jsonl")
    ap.add_argument("--hard", default="data/rag/hard_questions.jsonl")
    ap.add_argument("--negatives", default="data/rag/negative_questions.jsonl")
    ap.add_argument("--modes", default="dense,hybrid")
    ap.add_argument("--out", default="data/rag/metrics/retrieval.json")
    a = ap.parse_args()

    positives, hard, negatives = load(a.questions), load(a.hard) if a.hard else [], load(a.negatives)
    retriever = Retriever(use_query_cache=False)
    check_labels(retriever, positives, hard)

    metrics, per_mode = {}, {}
    for mode in a.modes.split(","):
        t0 = time.perf_counter()
        metrics[mode], per_mode[mode] = run_mode(retriever, positives, hard, negatives, **MODES[mode])
        n_queries = len(positives) + len(hard) + len(negatives)
        metrics[mode]["seconds_per_question"] = round((time.perf_counter() - t0) / n_queries, 3)

    print(f"{'mode':<8}{'group':<12}{'n':>4}{'hit@1':>8}{'hit@3':>8}{'hit@5':>8}{'MRR':>8}")
    for mode, m in metrics.items():
        for group, s in m.items():
            if isinstance(s, dict):
                print(f"{mode:<8}{group:<12}{s['n']:>4}{s['hit@1']:>8.2f}{s['hit@3']:>8.2f}{s['hit@5']:>8.2f}"
                      f"{s['mrr']:>8.2f}")
        ci = m["all"]
        print(f"{mode:<8}95% interval: hit@1 {ci['hit@1_ci95'][0]:.2f}-{ci['hit@1_ci95'][1]:.2f}, "
              f"hit@5 {ci['hit@5_ci95'][0]:.2f}-{ci['hit@5_ci95'][1]:.2f}")
        print(f"{mode:<8}negatives stopped by the score floor: {m['negatives_blocked_by_floor']:.0%}, "
              f"{m['seconds_per_question']}s/question\n")

    first = next(iter(per_mode.values()))
    misses = [r for r in first if not r["pos"]]
    print(f"{len(misses)} question(s) whose answer is not in the top 5 ({next(iter(per_mode))}):")
    for r in misses:
        print(f"  [{r['lang']}] {r['q'][:70]}  -> got {r['got'][:1]}")

    if len(per_mode) == 2:
        (m1, r1), (m2, r2) = per_mode.items()
        changed = [(x, y) for x, y in zip(r1, r2) if x["pos"] != y["pos"]]
        print(f"\n{len(changed)} question(s) ranked differently ({m1} -> {m2}, 0 = not in top 5):")
        for x, y in changed:
            better = y["pos"] and (not x["pos"] or y["pos"] < x["pos"])
            print(f"  {'better' if better else 'worse':<7}{x['pos']} -> {y['pos']}  [{x['lang']}] {x['q'][:70]}")

    out = ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nmetrics written to {a.out}")


if __name__ == "__main__":
    main()
