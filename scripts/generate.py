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
  python scripts/generate.py --model openai/gpt-oss-20b "question"
  python scripts/generate.py -k 5 --no-rerank "question"
"""
import argparse
import os
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from retriever import Retriever, Hit  # noqa: E402

DEFAULT_MODEL = "openai/gpt-oss-120b"

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
}
FAREWELL_REPLY = {
    "fr": "Avec plaisir ! N'hésite pas à revenir si tu as une autre question sur tes "
          "démarches. Bonne continuation !",
    "ar": "بكل سرور! لا تتردد في العودة إذا كان لديك سؤال آخر حول إجراءاتك. أراك قريباً!",
}

NO_MATCH_MESSAGE = {
    "fr": "Je n'ai pas d'information fiable sur ce sujet dans ma base de connaissances "
          "(démarches étudiantes et professionnelles en Tunisie). Je préfère ne pas répondre "
          "plutôt que de deviner.",
    "ar": "لا تتوفر لدي معلومة موثوقة حول هذا الموضوع في قاعدة معارفي (الإجراءات الطلابية "
          "والمهنية في تونس). أفضل عدم الإجابة على التخمين.",
}

SYSTEM_PROMPT = """Tu es l'assistant Khotwa, qui aide les étudiants et travailleurs tunisiens \
à comprendre les démarches administratives (études, emploi, CNSS...).

Règles strictes :
1. Réponds UNIQUEMENT à partir des extraits numérotés fournis ci-dessous. N'invente aucune \
condition, pièce, délai ou montant qui n'y figure pas.
2. Réponds dans la même langue que la question (français ou arabe).
3. Cite les sources utilisées avec leur numéro entre crochets dans le texte, ex: [1], [2]. \
N'ajoute PAS toi-même de section "Sources" récapitulative à la fin -- les liens officiels \
complets y sont ajoutés automatiquement après ta réponse.
4. Si les extraits fournis décrivent plusieurs variantes de la même procédure pour des \
secteurs, catégories ou publics différents (ex: secteur agricole / non agricole / pêche, \
salarié / non salarié / employeur) et que la question ne précise pas laquelle s'applique, \
NE CHOISIS PAS au hasard : pose une seule question de clarification courte, en listant les \
options distinctes trouvées dans les extraits.
5. Si les extraits ne répondent pas vraiment à la question posée, dis-le clairement plutôt \
que de forcer une réponse approximative.
6. Sois concis et concret : conditions, pièces à fournir, étapes, délais -- seulement ce qui \
est présent dans les extraits.
"""


def build_context(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, 1):
        blocks.append(
            f"[{i}] {h.title} ({h.org}) -- {h.domain}/{h.subtopic} [{h.lang}]\n"
            f"Source : {h.url}\n"
            f"{h.text}"
        )
    return "\n\n".join(blocks)


def detect_lang(text: str, lang: str | None = None) -> str:
    return lang or ("ar" if any("؀" <= c <= "ۿ" for c in text) else "fr")


SOURCES_LABEL = {"fr": "Sources officielles", "ar": "المصادر الرسمية"}


def sources_block(hits: list[Hit], q_lang: str) -> str:
    """A direct, clickable link per cited document -- the point is trust: the user can open
    the actual .gov.tn / .tn institutional page and check the answer against it themselves,
    not just take the model's word for it."""
    label = SOURCES_LABEL.get(q_lang, SOURCES_LABEL["fr"])
    lines = [f"[{i}] {h.title} ({h.org}) — {h.url}" for i, h in enumerate(hits, 1)]
    return f"\n\n**{label} :**\n" + "\n".join(lines)


def answer(question: str, retriever: Retriever | None = None, k: int = 3, lang: str | None = None,
           domain: str | None = None, subtopic: str | None = None, rerank: bool = True,
           model: str = DEFAULT_MODEL) -> dict:
    """Returns {"answer": str, "hits": list[Hit], "abstained": bool}.

    `retriever` is only needed past the chitchat check -- greetings/farewells are answered
    (and abstention on an empty retriever... no, see below) without ever loading the embedding
    model, the reranker, or requiring GROQ_API_KEY to be set. Pass an existing Retriever when
    calling this repeatedly (e.g. a server) to avoid reloading it per question; the CLI below
    builds one lazily since a one-shot process has nothing to reuse anyway."""
    q_lang = detect_lang(question, lang)

    if GREETING_RE.match(question):
        return {"answer": GREETING_REPLY.get(q_lang, GREETING_REPLY["fr"]), "hits": [], "abstained": False}
    if FAREWELL_RE.match(question):
        return {"answer": FAREWELL_REPLY.get(q_lang, FAREWELL_REPLY["fr"]), "hits": [], "abstained": False}

    retriever = retriever or Retriever()
    hits = retriever.search(question, k=k, lang=lang, domain=domain, subtopic=subtopic, rerank=rerank)
    if not hits:
        return {"answer": NO_MATCH_MESSAGE.get(q_lang, NO_MATCH_MESSAGE["fr"]), "hits": [], "abstained": True}

    if not os.environ.get("GROQ_API_KEY"):
        sys.exit("GROQ_API_KEY not set. Export it, or put GROQ_API_KEY=... in a .env file at the repo root.")
    from groq import Groq
    client = Groq()  # reads GROQ_API_KEY from the environment
    context = build_context(hits)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Extraits disponibles :\n\n{context}\n\nQuestion : {question}"},
        ],
        temperature=0.2,
    )
    text = resp.choices[0].message.content + sources_block(hits, q_lang)
    return {"answer": text, "hits": hits, "abstained": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--lang")
    ap.add_argument("--domain")
    ap.add_argument("--subtopic")
    ap.add_argument("--no-rerank", dest="rerank", action="store_false")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    a = ap.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    result = answer(a.question, k=a.k, lang=a.lang, domain=a.domain,
                     subtopic=a.subtopic, rerank=a.rerank, model=a.model)

    print(result["answer"])


if __name__ == "__main__":
    main()
