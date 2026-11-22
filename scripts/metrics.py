"""Precision@k / recall@k for a ranked retrieval list against a relevant-doc set.

Generic -- works on any list of retrieved ids plus the ground-truth relevant ids,
independent of the retriever used (BM25, dense, hybrid). Most RAG eval sets have
exactly one relevant doc per question, but these accept a set so a question with
several correct docs (duplicated content, multi-source answers) scores correctly.

    from metrics import precision_at_k, recall_at_k, average_precision

    precision_at_k(["a", "b", "c"], {"b"}, k=3)   # 1/3
    recall_at_k(["a", "b", "c"], {"b", "d"}, k=3)  # 1/2
"""


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of the top-k retrieved ids that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved[:k]
    return sum(1 for d in top_k if d in relevant) / len(top_k) if top_k else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of all relevant ids found within the top-k retrieved."""
    if not relevant:
        return 0.0
    hits = sum(1 for d in retrieved[:k] if d in relevant)
    return hits / len(relevant)


def average_precision(retrieved: list[str], relevant: set[str]) -> float:
    """Average precision: mean of precision@k taken at each rank a relevant id appears."""
    if not relevant:
        return 0.0
    hits, total = 0, 0.0
    for i, d in enumerate(retrieved, start=1):
        if d in relevant:
            hits += 1
            total += hits / i
    return total / len(relevant)


def mean_at_k(per_query: list[tuple[list[str], set[str]]], k: int) -> dict[str, float]:
    """Mean precision@k / recall@k / MAP over a batch of (retrieved, relevant) pairs."""
    n = len(per_query)
    if n == 0:
        return {"precision@k": 0.0, "recall@k": 0.0, "map": 0.0, "n": 0}
    p = sum(precision_at_k(r, rel, k) for r, rel in per_query) / n
    rec = sum(recall_at_k(r, rel, k) for r, rel in per_query) / n
    ap = sum(average_precision(r, rel) for r, rel in per_query) / n
    return {"precision@k": p, "recall@k": rec, "map": ap, "n": n}


if __name__ == "__main__":
    retrieved = ["a", "b", "c", "d", "e"]
    relevant = {"b", "d", "z"}
    for k in (1, 3, 5):
        print(f"k={k}  precision@k={precision_at_k(retrieved, relevant, k):.2f}  "
              f"recall@k={recall_at_k(retrieved, relevant, k):.2f}")
    print(f"average_precision={average_precision(retrieved, relevant):.2f}")
