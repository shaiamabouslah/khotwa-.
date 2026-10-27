"""Build the 3 chunking strategies (heading / fixed / semantic) side by side and score each
one with both evaluators (BM25 and BGE-M3 dense) on the same auto-generated questions, so the
numbers are directly comparable. Prints one summary table.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/compare_chunking.py
    PYTHONPATH=. .venv/Scripts/python.exe scripts/compare_chunking.py --file questions.jsonl
    PYTHONPATH=. .venv/Scripts/python.exe scripts/compare_chunking.py --strategies heading,fixed
        # skip semantic (no torch/sentence-transformers installed, or just faster)

Output chunk sets live in data/rag/cmp_<strategy>/ -- separate from data/rag/ itself (the
clean_pipeline.py --apply production output, always 'heading'), so comparing never touches it.
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HIT_RE = re.compile(r"(\d+) questions \| hit@1 ([\d.]+)\s+hit@3 ([\d.]+)\s+hit@5 ([\d.]+)\s+MRR ([\d.]+)")


def run(cmd, label):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if r.returncode:
        print(f"  FAILED: {label}\n{r.stderr}", file=sys.stderr)
        return None
    return r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb")
    ap.add_argument("--strategies", default="heading,fixed,semantic")
    ap.add_argument("--file", help="real questions file {q, doc_id, lang} per line -- beats --auto")
    ap.add_argument("--embed-model", default="BAAI/bge-m3")
    args = ap.parse_args()
    strategies = args.strategies.split(",")

    rows = []
    for strat in strategies:
        out_dir = ROOT / "data" / "rag" / f"cmp_{strat}"
        print(f"\n=== {strat}: building chunks -> {out_dir.relative_to(ROOT)} ===")
        cmd = [sys.executable, "scripts/rag_prepare.py", "--kb", args.kb, "--out", str(out_dir), "--strategy", strat]
        if strat == "semantic":
            cmd += ["--embed-model", args.embed_model]
        out = run(cmd, f"rag_prepare --strategy {strat}")
        if out is None:
            continue
        print(out.strip().split("\n")[2])  # the "chunks: N (...)" line

        chunks_path = out_dir / "chunks.jsonl"
        print(f"  BM25 eval...")
        bm25_cmd = [sys.executable, "scripts/fixes/eval_retrieval.py", "--chunks", str(chunks_path)]
        bm25_cmd += ["--file", args.file] if args.file else ["--auto"]
        bm25_out = run(bm25_cmd, f"eval_retrieval {strat}") or ""
        bm25_m = HIT_RE.search(bm25_out)

        print(f"  {args.embed_model} dense eval (first run downloads/embeds, can take a while)...")
        dense_cmd = [sys.executable, "scripts/fixes/eval_dense.py", "--chunks", str(chunks_path), "--embed-model", args.embed_model]
        dense_cmd += ["--file", args.file] if args.file else ["--auto"]
        dense_out = run(dense_cmd, f"eval_dense {strat}") or ""
        dense_m = HIT_RE.search(dense_out)

        rows.append((strat, bm25_m, dense_m))

    print("\n" + "=" * 78)
    print(f"{'strategy':<10} {'eval':<8} {'n':>5} {'hit@1':>7} {'hit@3':>7} {'hit@5':>7} {'MRR':>7}")
    print("-" * 78)
    for strat, bm25_m, dense_m in rows:
        for label, m in (("BM25", bm25_m), ("dense", dense_m)):
            if m:
                n, h1, h3, h5, mrr = m.groups()
                print(f"{strat:<10} {label:<8} {n:>5} {h1:>7} {h3:>7} {h5:>7} {mrr:>7}")
            else:
                print(f"{strat:<10} {label:<8}   FAILED (see stderr above)")
    print("=" * 78)


if __name__ == "__main__":
    main()
