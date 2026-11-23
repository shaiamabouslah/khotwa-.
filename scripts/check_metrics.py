"""Quality gate on the retrieval benchmark (scripts/benchmark_retrieval.py output): exits non-zero
when dense retrieval falls under the floors below, so a change to the chunking, the embeddings,
the knowledge base or the retrieval code cannot silently degrade what the assistant finds.

  python scripts/check_metrics.py data/rag/metrics/retrieval.json

The floors sit under the current scores by more than the benchmark's 95% interval width, so
noise does not fail a build but a real regression does.
"""
import json
import sys
from pathlib import Path

FLOORS = {
    ("all", "hit@5"): 0.92,
    ("all", "hit@1"): 0.72,
    ("all", "mrr"): 0.80,
}
PER_LANGUAGE_HIT5 = 0.80


def violations(metrics: dict) -> list[str]:
    dense = metrics["dense"]
    out = [f"{group} {name} = {dense[group][name]:.2f} < {floor:.2f}"
           for (group, name), floor in FLOORS.items() if dense[group][name] < floor]
    out += [f"{group} hit@5 = {s['hit@5']:.2f} < {PER_LANGUAGE_HIT5:.2f}"
            for group, s in dense.items()
            if isinstance(s, dict) and group not in ("all",) and s["hit@5"] < PER_LANGUAGE_HIT5]
    return out


def main():
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "data/rag/metrics/retrieval.json")
    failed = violations(json.loads(path.read_text(encoding="utf-8")))
    for v in failed:
        print(f"FAIL {v}")
    if failed:
        sys.exit(1)
    print("retrieval quality gate passed")


if __name__ == "__main__":
    main()
