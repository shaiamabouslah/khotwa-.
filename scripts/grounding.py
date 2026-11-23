"""Deterministic grounding check of a generated answer against the sources it was given.

For administrative procedures the most harmful hallucinations are concrete facts: an amount, a
deadline, an age limit, a phone number, a link. This check needs no model and no API call, so it
runs on every live answer (logged by the web app, shown on /metrics.html) and in the tests:

- every number of two digits or more in the answer must appear in the sources or the question
  (single digits are skipped: list numbering and "2 copies" vs "deux copies" make them noisy),
- every link in the answer must appear in the sources,
- every [n] citation must point to one of the n sources actually given.

It cannot judge a paraphrased claim -- that is what RAGAS faithfulness measures -- but a flagged
answer is a concrete, explainable signal: "this figure is in no source".
"""
import re

ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
NUMBER_RE = re.compile(r"\d+(?:[ .,  ]\d{3})*(?:[.,]\d+)?")
URL_RE = re.compile(r"https?://[^\s)\]>\"'*]+")
CITATION_RE = re.compile(r"\[(\d+)\]")
LIST_MARKER_RE = re.compile(r"(?m)^\s*\d+[.)]\s")


def _norm_number(raw: str) -> str:
    n = raw.translate(ARABIC_INDIC)
    n = re.sub(r"[   ]", "", n)
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", n):
        n = re.sub(r"[.,]", "", n)
    return n.replace(",", ".")


SCALE_RE = re.compile(r"\s*(ألف|آلاف|الف|mille|milles|thousand|k\b|مليون|ملايين|millions?)", re.I)
SCALES = {"ألف": 1_000, "آلاف": 1_000, "الف": 1_000, "mille": 1_000, "milles": 1_000, "thousand": 1_000,
          "k": 1_000, "مليون": 1_000_000, "ملايين": 1_000_000, "million": 1_000_000, "millions": 1_000_000}


def _scaled(n: str, scale_word: str) -> str:
    value = float(n) * SCALES[scale_word.lower()]
    return str(int(value)) if value.is_integer() else str(value)


def numbers(text: str) -> set[str]:
    """Numbers of 2+ digits; "75 ألف" / "75 mille" / "1,5 million" count as their full value."""
    text = CITATION_RE.sub(" ", LIST_MARKER_RE.sub(" ", text.translate(ARABIC_INDIC)))
    text = URL_RE.sub(" ", text)
    out = set()
    for m in NUMBER_RE.finditer(text):
        n = _norm_number(m.group())
        scale = SCALE_RE.match(text, m.end())
        if scale:
            n = _scaled(n, scale.group(1))
        if len(re.sub(r"\D", "", n)) >= 2:
            out.add(n)
    return out


def _norm_url(u: str) -> str:
    """The same page written as http/https, with or without www."""
    u = re.sub(r"^https?://(www\.)?", "", u.lower())
    return u.rstrip(".,;:").rstrip("/")


def check(answer: str, sources: list[str], question: str = "") -> dict:
    """answer: the generated text without the appended sources list; sources: the texts the LLM
    was given (title, url and text of each hit), in the order they were numbered."""
    reference = "\n".join(sources) + "\n" + question
    allowed_numbers = numbers(reference)
    # Phone numbers are written with varying groupings ("(216.71) 791 132" vs "71 791 132"):
    # long numbers are compared on their digits inside the same digit run of the sources.
    digit_runs = re.findall(r"\d+", re.sub(r"(?<=\d)[\s.()\-/]+(?=\d)", "", reference.translate(ARABIC_INDIC)))

    def supported(n: str) -> bool:
        digits = re.sub(r"\D", "", n)
        return n in allowed_numbers or (len(digits) >= 6 and any(digits in run for run in digit_runs))

    unsupported_numbers = sorted(n for n in numbers(answer) if not supported(n))
    allowed_urls = {_norm_url(u) for u in URL_RE.findall(reference)}
    unsupported_urls = sorted({u for u in URL_RE.findall(answer) if _norm_url(u) not in allowed_urls})
    bad_citations = sorted({int(n) for n in CITATION_RE.findall(answer) if not 1 <= int(n) <= len(sources)})
    return {
        "unsupported_numbers": unsupported_numbers,
        "unsupported_urls": unsupported_urls,
        "bad_citations": bad_citations,
        "grounded": not (unsupported_numbers or unsupported_urls or bad_citations),
    }


def source_texts(hits) -> list[str]:
    return [f"{h.title}\n{h.url}\n{h.text}" for h in hits]
