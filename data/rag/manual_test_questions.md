# Questions de test manuel — Khotwa

Jeu de questions pour tester le chatbot à la main (`python scripts/generate.py --chat`),
hors du pipeline RAGAS/retrieval automatisé. Couvre 4 catégories x FR/AR/EN :
salutations, procédures étudiants, procédures employés/employeurs, hors-contexte.

Comportement attendu entre parenthèses pour chaque catégorie.

## 1. Salutations (attendu : réponse de courtoisie, PAS de recherche dans la KB)

- Bonjour
- Salut, comment ça va ?
- مرحباً
- أهلا وسهلا
- Hello, how are you?
- Hi there

### Au revoir / remerciement (attendu : réponse de clôture, pas d'abstention)

- Merci beaucoup
- Merci, au revoir
- شكراً جزيلاً
- مع السلامة
- Thanks a lot, bye
- Thank you!

## 2. Procédures étudiants (`studies/*`) — attendu : réponse sourcée sur la KB

| Sous-thème | Question |
|---|---|
| equivalence | Comment faire reconnaître un diplôme étranger en Tunisie ? |
| equivalence | كيف يمكنني الحصول على معادلة شهادتي الأجنبية؟ |
| bourses | Quelles sont les conditions pour obtenir une bourse universitaire en tant que nouveau bachelier ? |
| bourses | ما هي شروط الحصول على منحة جامعية؟ |
| inscription | How do I request a university transfer (réorientation) ? |
| releves-attestations | Comment obtenir une attestation de réussite ou un duplicata de relevé de notes ? |
| legalisation | Quelles sont les étapes pour légaliser mon diplôme pour une utilisation à l'étranger ? |
| stages-pfe | ما هي الوثائق المطلوبة لاتفاقية تربص PFE؟ |
| mobilite | What are the conditions to apply for an Erasmus exchange program? |

## 3. Procédures employés / employeurs (`employment/*`) — attendu : réponse sourcée sur la KB

| Sous-thème | Question |
|---|---|
| affiliation-cnss | Comment un employeur déclare-t-il un salarié à la CNSS ? |
| affiliation-cnss | كيف ينخرط الأجير في الضمان الاجتماعي؟ |
| employes-de-maison | Quelles sont les démarches pour déclarer une employée de maison à la CNSS ? |
| non-salaries | How does a self-employed/independent worker affiliate with the CNSS? |
| patente-fiscalite | Comment obtenir une carte d'identification fiscale (patente) ? |
| attestations | كيف أحصل على شهادة انخراط أو كشف مساهمات من الضمان الاجتماعي؟ |
| creation-entreprise | Quelles sont les étapes pour créer une entreprise (SUARL) en Tunisie ? |
| creation-entreprise | ما هي خطوات بعث مؤسسة عبر السجل الوطني للمؤسسات؟ |
| recherche-emploi | What ANETI programs exist for job seekers (SIVP, stage d'initiation) ? |

## 4. "Nouveautés" / actualité (attendu : abstention probable — la KB documente des

procédures stables, pas des annonces récentes ; bon test de la limite déclarative du système)

- Y a-t-il de nouvelles mesures pour les bourses universitaires cette année ?
- Quels sont les derniers changements dans la réglementation CNSS ?
- ما هي آخر التحديثات بخصوص التأمين الاجتماعي هذا العام؟
- Is there a new version of the SIVP program in 2026?
- Est-ce que les frais de patente ont changé récemment ?

## 5. Hors-contexte (attendu : abstention claire, pas d'invention)

- Quelle est la recette du couscous tunisien ? (déjà testé — OK)
- Quelle est la météo à Tunis aujourd'hui ?
- من فاز بكأس العالم لكرة القدم؟
- Can you write me a Python script to sort a list?
- Comment obtenir un passeport en France ? (pays différent — piège : vérifier qu'il ne
  répond pas en confondant avec une procédure tunisienne)
- ما هو عاصمة اليابان؟
- What's the capital of Japan?
- Peux-tu m'aider à réviser pour un examen de maths ?

## 6. Cas limites (optionnel, plus difficile)

- Je voudrais des infos sur la CNSS (vague — entre salutation et vraie question,
  vérifier qu'il ne traite pas ça comme une salutation)
- C présiz les conditions pour la bourse étudiante ? (fautes/langage familier)
- Bonjour, comment faire pour la CNSS ? (salutation + question dans le même message)
