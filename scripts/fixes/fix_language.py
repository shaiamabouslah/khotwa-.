"""Problem 5: wrong `lang` labels. Only relabels when the text is clearly in another language;
bilingual forms (Arabic + French side by side) are left alone. Old value kept as lang_original."""
from _common import run, script_share
from crawler.classify import detect_lang


def fix(d):
    """language labels"""
    if d.fm.get("lang_original"): return None                          # relabel once, never flip back and forth
    ar, _ = script_share(d.body)
    cur = d.fm.get("lang")
    if cur in ("fr", "en") and ar > 0.6: new = "ar"
    elif cur == "ar" and ar < 0.1 and len(d.body) > 300: new = detect_lang(d.body)
    else: return None
    if new == cur: return None
    d.fm["lang_original"], d.fm["lang"] = cur, new
    return f"{cur} -> {new}"


if __name__ == "__main__": run("language", fix)
