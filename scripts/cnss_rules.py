"""CNSS-specific rules applied around retrieval and generation (scripts/generate.py).

The CNSS publishes the same procedure once per sector and once per role (employer, employee,
self-employed), as near-identical pages. Dense retrieval alone picks among them almost at random,
so these deterministic rules decide instead: ask for the sector when the question doesn't name
one, recognise the reply, and move pages written for another role to the back.
Supporting another administration with the same structure means extending the tables here.
"""
import unicodedata


# affiliation-cnss and non-salaries each bundle dozens of near-duplicate CNSS pages that only
# differ by the applicant's secteur (agricole / non agricole / peche / ...). Retrieval alone
# can't be trusted to resolve this: that many boilerplate-sized pages embed close together, so
# the top-k window surfaces a near-random subset instead of the full menu -- seen by hand:
# "comment m'affilier a la cnss" flipped between "secteur non agricole" and "secteur peche"
# pages depending on phrasing. Asking the LLM to "notice ambiguity and ask" only works if every
# variant happens to be in its context window, which isn't guaranteed. So this is resolved
# deterministically, before the LLM (or even the full top-k) is involved -- same pattern as
# greetings/farewells bypassing the LLM entirely.
SECTOR_KEYWORDS = {
    "agricole": ["agricole", "agriculture", "فلاحي", "الفلاحي", "فلاحة"],
    "non agricole": ["non agricole", "non-agricole", "غير فلاحي", "غير الفلاحي"],
    "pêche": ["pêche", "peche", "pêcheur", "صيد بحري", "البحري", "fishing", "fisher"],
    "gens de maison": ["gens de maison", "employé de maison", "employée de maison",
                        "femme de ménage", "عاملة منزلية", "عملة المنازل", "domestic"],
    "chantiers": ["chantier", "chantiers", "بناء", "ورشات", "construction"],
    "artiste": ["artiste", "فنان", "artist"],
    "étudiant": ["étudiant", "étudiante", "طالب", "student"],
}
# English words too: without them an English reply ("student") to the English clarification
# didn't count as naming a sector, and the same question was asked again.
SECTOR_KEYWORDS["agricole"] += ["agricultural", "agriculture", "farm"]
SECTOR_KEYWORDS["non agricole"] += ["non-agricultural", "non agricultural"]
SECTOR_LABELS = {
    "agricole": {"fr": "agricole", "ar": "فلاحي", "en": "agricultural"},
    "non agricole": {"fr": "non agricole", "ar": "غير فلاحي", "en": "non-agricultural"},
    "pêche": {"fr": "pêche", "ar": "صيد بحري", "en": "fishing"},
    "gens de maison": {"fr": "gens de maison (employé(e) de maison)", "ar": "عاملة منزلية",
                        "en": "domestic work"},
    "chantiers": {"fr": "chantiers / bâtiment", "ar": "ورشات البناء", "en": "construction sites"},
    "artiste": {"fr": "artiste", "ar": "فنان", "en": "artist"},
    "étudiant": {"fr": "étudiant", "ar": "طالب", "en": "student"},
}
# subtopic -> ordered sector keys that genuinely apply to it (from kb/employment/<subtopic>/).
SECTOR_SUBTOPICS = {
    "affiliation-cnss": ["agricole", "non agricole", "pêche", "gens de maison", "chantiers"],
    "non-salaries": ["agricole", "non agricole", "pêche", "artiste", "étudiant"],
}
SECTOR_CLARIFY_MESSAGE = {
    "fr": "Cette démarche CNSS dépend de votre secteur d'activité. De quel secteur s'agit-il : {options} ?",
    "ar": "يعتمد هذا الإجراء المتعلق بالضمان الاجتماعي على قطاع نشاطك. ما هو القطاع المعني: {options}؟",
    "en": "This CNSS procedure depends on your sector of activity. Which sector applies: {options}?",
}


# The sector question must only be asked when the question is actually about the CNSS -- not just
# because the nearest document happens to be a CNSS page. Found by hand with dense-only retrieval:
# "ما هي إجراءات تجديد جواز السفر التونسي؟" (passport renewal, out of scope) got a CNSS affiliation
# page as its top hit and was answered with "which sector do you work in?".
CNSS_TOPIC_KEYWORDS = ["cnss", "sécurité sociale", "securite sociale", "social security", "affili",
                       "immatricul", "cotisation", "الضمان الاجتماعي", "ضمان اجتماعي", "أنخرط",
                       "انخراط", "الصندوق الوطني"]


def _mentions_cnss(text: str) -> bool:
    low = text.lower()
    return any(kw in low for kw in CNSS_TOPIC_KEYWORDS)


def _mentions_sector(text: str, subtopic: str) -> bool:
    low = text.lower()
    return any(any(kw.lower() in low for kw in SECTOR_KEYWORDS[sector])
               for sector in SECTOR_SUBTOPICS.get(subtopic, []))


def clarify_options(text: str) -> list[str]:
    """The options listed in a SECTOR_CLARIFY_MESSAGE (any language), so the UI can offer them as
    buttons and answer() can recognise a reply to it. [] for any other text."""
    for template in SECTOR_CLARIFY_MESSAGE.values():
        prefix, _, suffix = template.partition("{options}")
        if text.startswith(prefix) and text.endswith(suffix) and len(text) > len(prefix) + len(suffix):
            return [o.strip() for o in text[len(prefix):len(text) - len(suffix)].split(", ") if o.strip()]
    return []


def _fold(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s.lower()) if not unicodedata.combining(c)).strip()


def _is_reply_to_clarification(question: str, history) -> bool:
    if not history:
        return False
    q = _fold(question)
    return len(q) >= 3 and any(q in _fold(o) or _fold(o) in q for o in clarify_options(history[-1][1]))


# CNSS pages are written per role -- the same procedure exists for the employer, the employee and
# the self-employed, marked in their titles ("(Employeur)", "(Salarie)", "Non Sal..."). Found by
# hand: "Comment m'affilier à la CNSS comme employeur dans le secteur non agricole ?" ranked the
# employer pages 6th-10th (dense retrieval) behind self-employed pages -- "m'affilier" pulls
# toward them -- so the LLM, which only sees the top 5, said it had no information. When a CNSS
# question names a role, pages explicitly written for a DIFFERENT role move to the back; unmarked
# pages keep their place (a first version moved matching pages to the front instead, which pushed
# down the Arabic CNSS pages -- their titles carry no role marker -- and cost one question on the
# 41-question benchmark). Checked in order: "non salarié" must win over "salarié", "غير أجير" over
# "أجير", "mes salariés" (said by an employer) over "salarié".
# Title markers: the readable titles written by scripts/cnss_titles.py ("... — employeur, ...",
# "... — المؤجر، ..."), plus the raw CNSS codes they replaced, in case an old build is served.
CNSS_ROLES = [
    (["employeur", "employer", "mon entreprise", "ma société", "mes employés", "mes salariés",
      "صاحب عمل", "مؤجر", "مشغل"], ["— employeur", "— المؤجر", "(employeur)"]),
    (["non salarié", "non-salarié", "non salarie", "indépendant", "independant", "independent",
      "self-employed", "غير أجير", "مستقل", "لحسابه الخاص"], ["— non-salarié", "— غير الأجير", "non sal"]),
    (["salarié", "salarie", "employee", "أجير"], ["— salarié", "— الأجير", "(salarie)", "(salarié)"]),
]
ROLE_POOL = 20


def _doc_role(title: str) -> int | None:
    low = title.lower()
    for i, (_, title_markers) in enumerate(CNSS_ROLES):
        if any(m in low for m in title_markers):
            return i
    return None


def _role_first(query: str, hits: list) -> list:
    if not hits or not _mentions_cnss(query):
        return hits
    low = query.lower()
    for i, (query_keywords, _) in enumerate(CNSS_ROLES):
        if any(kw in low for kw in query_keywords):
            keep = [h for h in hits if _doc_role(h.title) in (None, i)]
            return keep + [h for h in hits if h not in keep]
    return hits
