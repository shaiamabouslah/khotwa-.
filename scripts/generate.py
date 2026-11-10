#!/usr/bin/env python3
"""Generation layer: retriever hits -> grounded answer with citations, via the Groq API.

Pipeline:
  1. Retriever.search(question) -- reuses the calibrated confidence threshold (see
     retriever.DEFAULT_MIN_RERANK_SCORE) to abstain when nothing in the KB really matches.
  2. No hits -> a fixed "I don't have this information" message (fr/ar aware). No LLM call:
     the model never gets a chance to improvise an answer retrieval didn't support.
  3. Hits found -> the top-k chunks are handed to the LLM as numbered, cited context. The
     system prompt tells it to answer ONLY from that context, in the question's own language,
     and -- this is the part that matters here -- to ask a clarifying question instead of
     guessing if the candidates look like variants of the same procedure for different
     sectors/audiences and the question didn't say which one applies. That's a direct response
     to a real failure found by hand: "comment m'affilier a la cnss" with no sector mentioned
     flipped between the "secteur non agricole" and "secteur peche" CNSS pages depending on
     phrasing. Rather than hand-tune a sector-detection heuristic, the LLM reading the actual
     titles/content is better placed to notice the ambiguity and ask, the way a human clerk
     would ("vous travaillez dans quel secteur ?").

Needs GROQ_API_KEY, either exported or in a .env file at the repo root (gitignored already).

CLI:
  python scripts/generate.py "question"
  python scripts/generate.py --model openai/gpt-oss-120b "question"
  python scripts/generate.py -k 5 --no-rerank "question"
  python scripts/generate.py --chat     # multi-turn: remembers the conversation so far,
                                         # so a one-word reply to a clarifying question
                                         # ("quel secteur ?" -> "agricole") actually works
  python scripts/generate.py --chat --session etudiant1   # resumes that session next time
  python scripts/generate.py --chat --reset-session       # clears the session before starting
"""
import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import asdict
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from retriever import Retriever, Hit  # noqa: E402

# Compared openai/gpt-oss-20b against openai/gpt-oss-120b for both generation and judging
# (scripts/eval_ragas.py, 2026-11-03): mean faithfulness 0.837 vs 0.834 on 6 questions --
# no measurable quality loss -- and 20b uses far fewer tokens per call, which matters given
# Groq's free-tier rate limits (8000 TPM / 200k TPD) bit hard during that same testing.
DEFAULT_MODEL = "openai/gpt-oss-20b"

# Chitchat (greetings, thanks, goodbyes) never reaches retrieval or the LLM -- a canned reply
# only, so none of it can turn into a hallucinated answer, and none of it costs an
# embedding+rerank+API call. Two buckets (opening vs. closing) because the reply differs.
GREETING_RE = re.compile(
    r"^\s*(bonjour|bonsoir|salut+|coucou|hello+|hi|hey|"
    r"ahla+|اهلا|أهلا|أهلين|مرحبا|السلام عليكم|سلام)\W*\s*$", re.I)
FAREWELL_RE = re.compile(
    r"^\s*(merci(\s+beaucoup)?|thanks?(\s+you)?|thx|"
    r"au\s?revoir|a\s?bient[oô]t|bonne\s+journ[ée]e|bonne\s+soir[ée]e|bye+|see\s?you|"
    r"شكرا(\s+جزيلا)?|متشكر|مع السلامة|إلى اللقاء|الى اللقاء|يعطيك الصحة)\W*\s*$", re.I)
GREETING_REPLY = {
    "fr": "Bonjour ! Je suis Khotwa, un assistant pour les démarches administratives des "
          "étudiants et travailleurs en Tunisie (bourses, inscriptions, CNSS, création "
          "d'entreprise...). Pose-moi une question précise sur une de ces démarches.",
    "ar": "أهلاً بك! أنا خطوة، مساعد للإجراءات الإدارية للطلبة والعاملين في تونس (المنح، "
          "التسجيل، الضمان الاجتماعي، تأسيس المشاريع...). اطرح عليّ سؤالاً محدداً حول "
          "أحد هذه الإجراءات.",
    "en": "Hello! I'm Khotwa, an assistant for administrative procedures for students and "
          "workers in Tunisia (scholarships, enrollment, social security, starting a "
          "business...). Ask me a specific question about one of these procedures.",
}
FAREWELL_REPLY = {
    "fr": "Avec plaisir ! N'hésite pas à revenir si tu as une autre question sur tes "
          "démarches. Bonne continuation !",
    "ar": "بكل سرور! لا تتردد في العودة إذا كان لديك سؤال آخر حول إجراءاتك. أراك قريباً!",
    "en": "You're welcome! Feel free to come back if you have another question about your "
          "procedures. Take care!",
}

BUSY_MESSAGE = {
    "fr": "Le service est momentanément surchargé (limite de requêtes atteinte). "
          "Merci de réessayer dans une minute.",
    "ar": "الخدمة مزدحمة حالياً (تم بلوغ الحد الأقصى للطلبات). يرجى المحاولة مجدداً بعد دقيقة.",
    "en": "The service is temporarily busy (rate limit reached). Please try again in a minute.",
}

NO_MATCH_MESSAGE = {
    "fr": "Désolé, je n'ai pas d'information fiable sur ce sujet. Je suis un assistant "
          "spécialisé : je réponds uniquement aux questions sur les démarches administratives "
          "des étudiants et travailleurs en Tunisie (bourses, inscriptions, CNSS, création "
          "d'entreprise...). Je préfère ne pas répondre plutôt que de deviner.",
    "ar": "آسف، لا تتوفر لدي معلومة موثوقة حول هذا الموضوع. أنا مساعد متخصص: أجيب فقط عن "
          "الأسئلة المتعلقة بالإجراءات الإدارية للطلبة والعاملين في تونس (المنح، التسجيل، "
          "الضمان الاجتماعي، تأسيس المشاريع...). أفضل عدم الإجابة على التخمين.",
    "en": "Sorry, I don't have reliable information on this topic. I'm a specialized "
          "assistant: I only answer questions about administrative procedures for students "
          "and workers in Tunisia (scholarships, enrollment, social security, starting a "
          "business...). I'd rather not answer than guess.",
}

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
    "pêche": ["pêche", "peche", "pêcheur", "صيد بحري", "البحري"],
    "gens de maison": ["gens de maison", "employé de maison", "employée de maison",
                        "femme de ménage", "عاملة منزلية", "عملة المنازل"],
    "chantiers": ["chantier", "chantiers", "بناء", "ورشات"],
    "artiste": ["artiste", "فنان"],
    "étudiant": ["étudiant", "étudiante", "طالب"],
}
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


def _mentions_sector(text: str, subtopic: str) -> bool:
    low = text.lower()
    return any(any(kw.lower() in low for kw in SECTOR_KEYWORDS[sector])
               for sector in SECTOR_SUBTOPICS.get(subtopic, []))


SYSTEM_PROMPT = """Tu es l'assistant Khotwa, qui aide les étudiants et travailleurs tunisiens \
à comprendre les démarches administratives (études, emploi, CNSS...).

Règles strictes :
1. Réponds UNIQUEMENT à partir des extraits numérotés fournis ci-dessous. N'invente aucune \
condition, pièce, délai ou montant qui n'y figure pas.
2. Réponds dans la même langue que la question (français, arabe, ou anglais).
3. Cite les sources utilisées avec leur numéro entre crochets dans le texte, ex: [1], [2]. \
N'ajoute PAS toi-même de section "Sources" récapitulative à la fin -- les liens officiels \
complets y sont ajoutés automatiquement après ta réponse.
4. Si les extraits fournis décrivent plusieurs variantes de la même procédure pour des \
secteurs, catégories ou publics différents (ex: secteur agricole / non agricole / pêche, \
salarié / non salarié / employeur) et que la question ne précise pas laquelle s'applique, \
NE CHOISIS PAS au hasard : pose une seule question de clarification courte, en listant les \
options distinctes trouvées dans les extraits. Si en revanche la question actuelle est une \
réponse courte à une clarification QUE TU AS TOI-MÊME demandée au tour précédent (ex: tu as \
demandé "quel secteur ?" et l'utilisateur répond juste "agricole"), ne repose PAS la même \
question : utilise cette réponse pour choisir directement la bonne variante parmi les extraits \
et réponds à la question d'origine pour ce cas précis.
5. Si les extraits ne répondent pas vraiment à la question posée, dis-le clairement plutôt \
que de forcer une réponse approximative.
6. Sois concis et concret : conditions, pièces à fournir, étapes, délais -- seulement ce qui \
est présent dans les extraits.
"""


GEN_CACHE_PATH = ROOT / "data" / "rag" / "generation_cache.json"
HISTORY_DIR = ROOT / "data" / "rag" / "conversations"


def _history_path(session: str) -> Path:
    # session is user-supplied (CLI flag); keep it to safe filename characters so it can't
    # escape HISTORY_DIR or collide with an unrelated file.
    safe = re.sub(r"[^A-Za-z0-9_-]", "_", session) or "default"
    return HISTORY_DIR / f"{safe}.json"


def load_history(session: str) -> list[tuple[str, str, bool]]:
    path = _history_path(session)
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    # old files saved before the abstained flag was tracked are 2-element -- treat as grounded
    return [(row[0], row[1], row[2] if len(row) > 2 else False) for row in rows]


def save_history(session: str, history: list[tuple[str, str, bool]]):
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    _history_path(session).write_text(
        json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def _gen_cache_sig() -> str:
    """Invalidates the whole cache if either the chunks the answer was grounded in, or the
    system prompt that shaped how it was written, changed since the cache was built -- same
    idea as retriever.py's query cache, extended to also cover the prompt."""
    chunks_path = ROOT / "data" / "rag" / "chunks.jsonl"
    st = chunks_path.stat()
    prompt_hash = hashlib.sha1(SYSTEM_PROMPT.encode("utf-8")).hexdigest()[:12]
    return f"{st.st_size}:{st.st_mtime_ns}:{prompt_hash}"


def _gen_cache_key(question: str, k: int, lang, domain, subtopic, rerank: bool, model: str) -> str:
    norm_q = re.sub(r"\s+", " ", question.strip().lower())
    raw = "|".join(str(x) for x in (norm_q, k, lang, domain, subtopic, rerank, model))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def _load_gen_cache() -> dict:
    if not GEN_CACHE_PATH.exists():
        return {}
    try:
        saved = json.loads(GEN_CACHE_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return saved.get("entries", {}) if saved.get("sig") == _gen_cache_sig() else {}


def _save_gen_cache(entries: dict):
    GEN_CACHE_PATH.write_text(
        json.dumps({"sig": _gen_cache_sig(), "entries": entries}, ensure_ascii=False), encoding="utf-8")


def _call_groq_with_retry(client, max_retries: int = 1, **kwargs):
    """Groq's free tier has a tight per-minute token budget that gets hit routinely under
    any real testing load (seen repeatedly: TPM 429s that resolve in a few seconds). One
    short, bounded retry turns those into a silent success instead of a crash. A daily-cap
    429 reports a retry-after of many minutes -- not worth blocking the user for, so this
    only retries when the wait is short; anything longer is re-raised immediately for the
    caller to degrade gracefully instead of hanging."""
    import time
    from groq import RateLimitError

    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError as e:
            retry_after = None
            try:
                retry_after = float(e.response.headers.get("retry-after", ""))
            except (TypeError, ValueError, AttributeError):
                pass
            if attempt < max_retries and retry_after is not None and retry_after <= 15:
                time.sleep(retry_after + 0.5)
                continue
            raise


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] {h.title} ({h.org}) -- {h.domain}/{h.subtopic} [{h.lang}]\n"
            f"Source : {h.url}\n"
            f"{h.text}"
        )
    return "\n\n".join(blocks)


_FR_STOPWORDS = {"le", "la", "les", "des", "du", "et", "pour", "est", "une", "dans", "aux", "à", "comment", "quelle", "quelles", "quel"}
_EN_STOPWORDS = {"the", "and", "for", "is", "of", "to", "with", "your", "are", "this", "how", "what", "do", "does"}


def detect_lang(text: str, lang: str | None = None) -> str:
    """fr/en/ar, same heuristic as crawler/classify.py's detect_lang() (Arabic by script
    ratio, then French vs English by stopword overlap) so corpus-side and generation-side
    language detection agree."""
    if lang:
        return lang
    latin = re.findall(r"[A-Za-zÀ-ÿ]+", text)
    if sum(1 for c in text if "؀" <= c <= "ۿ") > sum(len(w) for w in latin):
        return "ar"
    words = {w.lower() for w in latin}
    return "en" if len(words & _EN_STOPWORDS) > len(words & _FR_STOPWORDS) else "fr"


SOURCES_LABEL = {"fr": "Sources officielles", "ar": "المصادر الرسمية", "en": "Official sources"}

# A per-call reminder, written IN the target language, appended after the retrieved context.
# The system prompt already says "answer in the question's language" once, in French -- that
# alone wasn't enough: with an EN question but 100% French source chunks, the model drifted
# into French anyway (seen by hand). A directive in the target language itself, right next to
# the question, is a much stronger signal than one generic French rule stated far upstream.
LANG_DIRECTIVE = {
    "fr": "Réponds à la question ci-dessus en français, même si certains extraits sont dans une autre langue.",
    "ar": "أجب عن السؤال أعلاه باللغة العربية، حتى لو كانت بعض المقتطفات بلغة أخرى.",
    "en": "Answer the question above in English, even if some of the excerpts are in another language.",
}


def sources_block(hits: list[Hit], q_lang: str) -> str:
    """A direct, clickable link per cited document -- the point is trust: the user can open
    the actual .gov.tn / .tn institutional page and check the answer against it themselves,
    not just take the model's word for it."""
    label = SOURCES_LABEL.get(q_lang, SOURCES_LABEL["fr"])
    lines = [f"[{i}] {h.title} ({h.org}) — {h.url}" for i, h in enumerate(hits, 1)]
    return f"\n\n**{label} :**\n" + "\n".join(lines)


def answer(question: str, retriever: Retriever | None = None, k: int = 5, lang: str | None = None,
           domain: str | None = None, subtopic: str | None = None, rerank: bool = True,
           model: str = DEFAULT_MODEL, use_cache: bool = True,
           history: list[tuple[str, str, bool]] | None = None) -> dict:
    """Returns {"answer": str, "hits": list[Hit], "abstained": bool}.

    `retriever` is only needed past the chitchat check -- greetings/farewells are answered
    (and abstention on an empty retriever... no, see below) without ever loading the embedding
    model, the reranker, or requiring GROQ_API_KEY to be set. Pass an existing Retriever when
    calling this repeatedly (e.g. a server) to avoid reloading it per question; the CLI below
    builds one lazily since a one-shot process has nothing to reuse anyway.

    use_cache: a repeat of the exact same (question, k, lang, domain, subtopic, rerank, model)
    call is served from data/rag/generation_cache.json without touching the retriever or Groq
    -- unlike retriever.py's own query cache, this also skips the LLM call itself, which is
    the point: Groq's free tier has tight per-minute and per-day token caps (hit repeatedly
    while testing), so a question asked twice shouldn't burn budget twice. Ignored when
    `history` is given: a multi-turn answer depends on the conversation so far, not just the
    latest message, and the cache key doesn't account for that.

    history: prior (user_text, assistant_text, abstained) turns, oldest first. Retrieval always
    tries the bare `question` first -- most follow-ups are complete questions on their own and
    combining them with the previous turn unconditionally was found (by hand) to actively hurt:
    a 3rd/4th turn would get glued to whatever the immediately preceding turn happened to be,
    including a prior abstention, and the query would drift off-topic turn by turn. Combining
    is now only a fallback, triggered when the bare question retrieves nothing.

    The fallback anchors on ALL non-abstained prior user turns concatenated (oldest first), not
    just the last one -- checked by hand with retriever.search(min_score=None) on a real 4-turn
    CNSS conversation: anchoring only on the immediately preceding turn ("agricole" alone, a
    one-word reply) gave a combined query that scored 0.15, well under the 0.3 threshold, for a
    later "quel est le delai ?" follow-up -- "agricole" alone just doesn't carry enough signal.
    Anchoring on the *original* question ("Comment affilier un travailleur independant a la
    CNSS ?") instead scored 0.96-0.98 for the same follow-up. The original question is usually
    where the actual topic/entities live; a short clarifying reply like "agricole" narrows it
    but rarely restates it, so dropping the original from the anchor loses most of the signal.
    This is still what makes a one-word reply like "agricole" to "vous travaillez dans quel
    secteur ?" resolve correctly (it has no signal alone, so the fallback kicks in with the
    accumulated history as context)."""
    q_lang = detect_lang(question, lang)

    if GREETING_RE.match(question):
        return {"answer": GREETING_REPLY.get(q_lang, GREETING_REPLY["fr"]), "hits": [], "abstained": False}
    if FAREWELL_RE.match(question):
        return {"answer": FAREWELL_REPLY.get(q_lang, FAREWELL_REPLY["fr"]), "hits": [], "abstained": False}

    use_cache = use_cache and not history
    cache_key = _gen_cache_key(question, k, lang, domain, subtopic, rerank, model) if use_cache else None
    cache = _load_gen_cache() if use_cache else {}
    if cache_key in cache:
        entry = cache[cache_key]
        return {"answer": entry["answer"], "hits": [Hit(**h) for h in entry["hits"]], "abstained": entry["abstained"]}

    retriever = retriever or Retriever()
    retrieval_query = question
    # `lang` here means "answer in this language" (-> detect_lang above), NOT "restrict
    # candidates to documents written in this language" -- the KB has zero English chunks,
    # so forwarding lang="en" into retriever.search()'s document-language filter would
    # silently exclude every French/Arabic doc an English query is actually meant to match
    # cross-lingually via BGE-M3. That was a real bug here (found while adding EN test
    # questions: every one abstained, not on merit, just because of this filter).
    hits = retriever.search(retrieval_query, k=k, domain=domain, subtopic=subtopic, rerank=rerank)
    # The anchor fallback below is for a short, ambiguous follow-up ("agricole") that only
    # makes sense combined with what was asked before -- NOT for a genuinely new, complete,
    # unrelated question that simply has no match in the KB. Without the word-count guard, a
    # real out-of-context question (e.g. "Quelle est la recette du couscous tunisien ?") that
    # legitimately finds zero hits on its own got the entire prior conversation glued onto it
    # as a retrieval query, dragged in unrelated chunks from earlier topics, and sent the LLM a
    # message history it then "recapped" turn by turn instead of cleanly abstaining on the
    # actual question asked (seen by hand: it answered 4 unrelated previous questions plus the
    # new one in one numbered list, at one point even reinventing a wrong answer for one of
    # them). A short question has no real content to go on by itself, so only it gets the
    # history-anchored retry; anything longer is trusted to stand on its own.
    if not hits and history and len(question.split()) <= 3:
        anchor = " ".join(h[0] for h in history if not h[2])
        if anchor:
            retrieval_query = f"{anchor} {question}"
            hits = retriever.search(retrieval_query, k=k, domain=domain, subtopic=subtopic, rerank=rerank)
    if hits and hits[0].subtopic in SECTOR_SUBTOPICS and not _mentions_sector(retrieval_query, hits[0].subtopic):
        options = ", ".join(SECTOR_LABELS[s][q_lang] for s in SECTOR_SUBTOPICS[hits[0].subtopic])
        msg = SECTOR_CLARIFY_MESSAGE.get(q_lang, SECTOR_CLARIFY_MESSAGE["fr"]).format(options=options)
        result = {"answer": msg, "hits": hits, "abstained": False}
    elif not hits:
        result = {"answer": NO_MATCH_MESSAGE.get(q_lang, NO_MATCH_MESSAGE["fr"]), "hits": [], "abstained": True}
    else:
        if not os.environ.get("GROQ_API_KEY"):
            sys.exit("GROQ_API_KEY not set. Export it, or put GROQ_API_KEY=... in a .env file at the repo root.")
        from groq import Groq, RateLimitError
        client = Groq()  # reads GROQ_API_KEY from the environment
        context = build_context(hits)
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for prev_q, prev_a, _ in (history or []):
            messages.append({"role": "user", "content": prev_q})
            messages.append({"role": "assistant", "content": prev_a})
        # On a follow-up turn, `question` alone can be a single word ("agricole") that only
        # makes sense combined with what was asked before -- send the same combined text the
        # retriever used, not the bare reply, or the model re-asks the question it just asked
        # instead of recognizing the reply as the answer to it (seen by hand: it did exactly
        # that when sent "agricole" alone, even with the right doc already in context).
        directive = LANG_DIRECTIVE.get(q_lang, LANG_DIRECTIVE["fr"])
        messages.append({"role": "user", "content":
                          f"Extraits disponibles :\n\n{context}\n\nQuestion : {retrieval_query}\n\n{directive}"})
        try:
            resp = _call_groq_with_retry(client, model=model, temperature=0.2, messages=messages)
        except RateLimitError:
            # Don't cache this -- it's a transient failure of the service, not a real answer
            # to "question", so a repeat of the same question a minute later must try again.
            return {"answer": BUSY_MESSAGE.get(q_lang, BUSY_MESSAGE["fr"]), "hits": [], "abstained": False}
        text = resp.choices[0].message.content + sources_block(hits, q_lang)
        result = {"answer": text, "hits": hits, "abstained": False}

    if use_cache:
        cache[cache_key] = {"answer": result["answer"], "hits": [asdict(h) for h in result["hits"]],
                             "abstained": result["abstained"]}
        _save_gen_cache(cache)
    return result


def chat_loop(retriever: Retriever, a):
    """Interactive multi-turn REPL. History is loaded from, and saved after every turn to,
    data/rag/conversations/<session>.json -- so closing the terminal (or a crash) doesn't
    throw away the conversation the way the in-memory-only version did. Saving after every
    turn rather than only at exit is deliberate: a crash mid-session should lose at most the
    turn in progress, not everything before it."""
    session = a.session
    history = [] if a.reset_session else load_history(session)
    if history:
        print(f"Khotwa -- mode conversation (session '{session}', {len(history)} tour(s) "
              "repris ; Ctrl+C ou 'exit' pour quitter)\n")
    else:
        print(f"Khotwa -- mode conversation (session '{session}' ; Ctrl+C ou 'exit' pour quitter)\n")
    while True:
        try:
            question = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question.lower() in ("exit", "quit", "q"):
            break
        result = answer(question, retriever, k=a.k, lang=a.lang, domain=a.domain,
                         subtopic=a.subtopic, rerank=a.rerank, model=a.model,
                         use_cache=False, history=history)
        print(result["answer"], "\n")
        history.append((question, result["answer"], result["abstained"]))
        save_history(session, history)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="?", help="omit when using --chat")
    ap.add_argument("-k", type=int, default=5,
                     help="top-k chunks sent to the LLM (default 5 -- 3 was too tight for "
                          "cross-lingual matches: the cross-encoder reranker under-scores a "
                          "genuinely relevant doc in a different language from the question, "
                          "so a strong candidate can fall just past a k=3 cutoff)")
    ap.add_argument("--lang")
    ap.add_argument("--domain")
    ap.add_argument("--subtopic")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-cache", dest="use_cache", action="store_false",
                     help="don't read/write data/rag/generation_cache.json for this call")
    ap.add_argument("--chat", action="store_true",
                     help="interactive multi-turn mode -- remembers the conversation so far")
    ap.add_argument("--session", default="default",
                     help="--chat only: name of the conversation to resume/save (default: 'default')")
    ap.add_argument("--reset-session", action="store_true",
                     help="--chat only: start the named session fresh instead of resuming it")
    a = ap.parse_args()
    if not a.chat and not a.question:
        sys.exit("a question is required unless --chat is used")

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    if a.chat:
        chat_loop(Retriever(), a)
        return

    result = answer(a.question, k=a.k, lang=a.lang, domain=a.domain,
                     subtopic=a.subtopic, rerank=a.rerank, model=a.model, use_cache=a.use_cache)

    print(result["answer"])


if __name__ == "__main__":
    main()
