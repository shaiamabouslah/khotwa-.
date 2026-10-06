"""Rule-based hints: language, document type, domain/subtopic.

These are *hints* stored in the corpus front matter. The cleaning stage can keep, fix or
replace them (e.g. with the router models); nothing downstream trusts the folder layout.
"""
import re
from dataclasses import dataclass

from .extract import normalize


def detect_lang(text: str) -> str:
    sample = text[:4000]
    arabic = len(re.findall(r"[؀-ۿ]", sample))
    latin = re.findall(r"[A-Za-zÀ-ÿ]+", sample)
    if arabic > sum(map(len, latin)):
        return "ar"
    words = {w.lower() for w in latin}
    fr = len(words & {"le", "la", "les", "des", "du", "et", "pour", "est", "une", "dans", "aux", "à"})
    en = len(words & {"the", "and", "for", "is", "of", "to", "with", "your", "are", "this"})
    return "en" if en > fr else "fr"


@dataclass
class Hint:
    domain: str | None
    subtopic: str | None
    score: int
    matched: list[str]


class Classifier:
    def __init__(self, taxonomy: dict):
        self.rules = []   # (domain, subtopic, normalized keyword)
        for domain, d in taxonomy["domains"].items():
            for sub, s in d["subtopics"].items():
                for kw in s.get("keywords", []):
                    self.rules.append((domain, sub, normalize(kw)))

    def classify(self, title: str, breadcrumb: list[str], body: str, site_domain: str | None = None) -> Hint:
        """Keywords in the title/breadcrumb weigh 3, in the first 400 words weigh 1.
        The site's declared domain (if any) breaks ties and filters cross-domain noise."""
        head = normalize(" ".join([title, *breadcrumb]))
        text = normalize(" ".join(body.split()[:400]))
        scores: dict[tuple[str, str], int] = {}
        matched: dict[tuple[str, str], list[str]] = {}
        for domain, sub, kw in self.rules:
            pattern = rf"(?<!\w){re.escape(kw)}(?!\w)"
            s = 3 * len(re.findall(pattern, head)) + len(re.findall(pattern, text))
            if s:
                scores[(domain, sub)] = scores.get((domain, sub), 0) + s
                matched.setdefault((domain, sub), []).append(kw)
        if not scores:
            return Hint(site_domain, None, 0, [])
        best = max(scores, key=lambda k: (scores[k], k[0] == site_domain))
        if site_domain and best[0] != site_domain:
            # A site of one domain rarely publishes the other: the other domain must clearly win.
            own = {k: v for k, v in scores.items() if k[0] == site_domain}
            if own:
                best_own = max(own, key=own.get)
                if scores[best] < 2 * own[best_own]:
                    best = best_own
        return Hint(best[0], best[1], scores[best], matched[best])


def doc_type(url: str, title: str, is_pdf: bool, rules: list[dict]) -> str:
    """procedure | announcement | form | guide | page  (site rules first, then defaults)."""
    for rule in rules or []:
        if re.search(rule["match"], url, re.I) or re.search(rule["match"], title, re.I):
            return rule["type"]
    t = normalize(title)
    if is_pdf:
        if re.search(r"guide|manuel|دليل", t):
            return "guide"
        return "form" if re.search(r"demande|formulaire|imprim|مطلب|مطبوع|استمارة", t) else "document"
    if re.search(r"^(communiqu|avis|بلاغ|إعلان|اعلان)", t):
        return "announcement"
    return "page"
