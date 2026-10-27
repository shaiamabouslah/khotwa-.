"""Run every fix in order, repeating (with --apply) until a pass changes nothing, because fixes can expose
each other's work (e.g. links -> plain text -> nav line).   python scripts/fixes/run_all.py [--apply] [--site cnss]
Order: unreadable -> layout_junk -> language -> links -> doc_types -> titles -> freshness -> catalog."""
import re, subprocess, sys
from pathlib import Path

ORDER = ["unreadable", "layout_junk", "language", "links", "doc_types", "titles", "freshness", "catalog"]


def one_pass():
    total = 0
    for name in ORDER:
        r = subprocess.run([sys.executable, str(Path(__file__).parent / f"fix_{name}.py"), *sys.argv[1:]], capture_output=True, text=True)
        if r.returncode: sys.exit(r.stderr or f"fix_{name} failed")
        print(r.stdout, end=""); total += int(re.search(r": (\d+)/", r.stdout)[1])
    return total


for i in range(4 if "--apply" in sys.argv else 1):
    print(f"-- pass {i + 1}")
    if one_pass() == 0 or "--apply" not in sys.argv: break
