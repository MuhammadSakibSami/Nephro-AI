# Nephro-AI — LLM অংশ: সম্পূর্ণ ইন্টারনাল ডকুমেন্টেশন (বাংলা)

> **এই ফাইলটির উদ্দেশ্য।** টিমের নিজেদের জন্য একটি রেফারেন্স ডকুমেন্ট — পড়ে *বোঝার* জন্য লেখা, বাইরের কারও জন্য পালিশ করা deliverable হিসেবে নয়। এই প্রজেক্টের LLM অংশ সম্পর্কে সবকিছু এখানে আছে: এটা কী করে, কেন আছে, কোন কোন technology ব্যবহার করা হয়েছে এবং কেন, প্রতিটা অংশ কীভাবে বানানো হয়েছে, আমরা কী কী experiment চালিয়েছি, এবং কী কী result পেয়েছি (যেগুলো আমাদের পক্ষে যায়নি সেগুলোসহ)।
>
> **সঙ্গের ফাইল:** `docs/LLM_DOCUMENTATION_EN.md` (একই কনটেন্ট ইংরেজিতে)।
>
> সর্বশেষ আপডেট: Q1 six-arm final evaluation পর্যন্ত।

---

## সূচিপত্র

1. [সহজ ভাষায় পরিচিতি](#১-সহজ-ভাষায়-পরিচিতি)
2. [সিস্টেমে LLM-এর অবস্থান](#২-সিস্টেমে-llm-এর-অবস্থান)
3. [LLM কেন লাগল, আর এত কড়া নিয়ন্ত্রণ কেন](#৩-llm-কেন-লাগল-আর-এত-কড়া-নিয়ন্ত্রণ-কেন)
4. [ব্যবহৃত প্রতিটি technology এবং কারণ](#৪-ব্যবহৃত-প্রতিটি-technology-এবং-কারণ)
5. [Output contract](#৫-output-contract)
6. [Phase ধরে ধরে পুরো নির্মাণ](#৬-phase-ধরে-ধরে-পুরো-নির্মাণ)
7. [সব experiment ও result](#৭-সব-experiment-ও-result)
8. [আমরা আসলে কী শিখলাম](#৮-আমরা-আসলে-কী-শিখলাম)
9. [সীমাবদ্ধতা ও সৎ স্বীকারোক্তি](#৯-সীমাবদ্ধতা-ও-সৎ-স্বীকারোক্তি)
10. [Reproducibility — command সমূহ](#১০-reproducibility--command-সমূহ)
11. [File map](#১১-file-map)
12. [পরিভাষা](#১২-পরিভাষা)

---

## ১. সহজ ভাষায় পরিচিতি

### পুরো প্রজেক্টটা কী করে

Nephro-AI-এর **দুইটা অংশ**:

1. **একটা machine-learning screener** (Final Model C) — ১৪টা সাধারণ স্বাস্থ্য তথ্য নেয় (বয়স, রক্তচাপ, রক্তে শর্করা, কিছু নিয়মিত রক্ত পরীক্ষা, আর কয়েকটা হ্যাঁ/না ধরনের history প্রশ্ন) এবং একটা kidney-risk category দেয়: **Low / Moderate / High**। এটা ইচ্ছাকৃতভাবে creatinine, eGFR বা UACR ব্যবহার করে **না** — অর্থাৎ যেসব পরীক্ষা দিয়ে সাধারণত কিডনি রোগ নির্ণয় করা হয় সেগুলো বাদ, কারণ আমাদের পুরো উদ্দেশ্যই হলো যাদের এই পরীক্ষাগুলো করা হয়নি তাদের screening করা।

2. **একটা LLM communication layer** (এই ডকুমেন্টের বিষয়) — ওই Low/Moderate/High ফলাফলকে রোগীর জন্য **নিরাপদ, ব্যক্তিগতকৃত, সহজবোধ্য স্বাস্থ্যশিক্ষায়** রূপান্তর করে।

### এক বাক্যে LLM-টা কী

> এটা একজন **অনুবাদক ও ব্যাখ্যাকারী**, ডাক্তার নয়।

ঝুঁকির মাত্রা ঠিক করে ML model। LLM-এর একমাত্র কাজ হলো সেই ফলাফলটা মানুষটাকে সহজ ভাষায় বুঝিয়ে বলা — তার নিজের অবস্থার সাথে মিলিয়ে, আসল clinical guideline-এর ভিত্তিতে, এবং কখনোই ডাক্তারি প্র্যাকটিসের সীমা না পেরিয়ে।

### একটা উপমা

ধরুন একজন ল্যাব টেকনিশিয়ান আপনাকে একটা রিপোর্ট ধরিয়ে দিল যেখানে লেখা "moderate risk"। শুধু এইটুকু রোগীর জন্য প্রায় অর্থহীন — এর মানে কী? এখন কী করতে হবে?

এবার কল্পনা করুন আপনার পাশে একজন **সতর্ক স্বাস্থ্য-শিক্ষক** বসে আছেন, যিনি:
- আপনার আসল সংখ্যাগুলো পড়েন (আপনার রক্তচাপ ১৫০, HbA1c ৭.৬),
- অফিসিয়াল kidney guideline-এর প্রাসঙ্গিক পাতাগুলো দেখে নেন,
- আপনার ফলাফলের মানে দৈনন্দিন ভাষায় ব্যাখ্যা করেন (প্রায় ক্লাস ৮ পড়ার মানে),
- খাবার, ঘুম, দৈনন্দিন রুটিন, আর কখন ডাক্তার দেখাতে হবে — সে বিষয়ে বাস্তব পরামর্শ দেন,
- এবং **কখনোই** বলেন না যে আপনার রোগ হয়েছে, কোনো ওষুধের নাম বলেন না, কোনো ডোজ দেন না।

ওই শিক্ষকই হলো LLM। আর যেহেতু AI শিক্ষক পথ হারাতে পারে, আমরা তার পেছনে একটা **কঠোর rule-based checker** বসিয়েছি, যেটা রোগীর কাছে পৌঁছানোর আগে প্রতিটা উত্তর পরীক্ষা করে এবং অনিরাপদ কিছু থাকলে ঠিক করে দেয় বা আটকে দেয়।

### তিনটি কঠিন নিয়ম

LLM কখনোই পারবে না:
1. **রোগ নির্ণয় করতে।** "আপনার chronic kidney disease আছে" — এটা বলা যাবে না। Screening ফলাফল একটা ঝুঁকির মাত্রা, রোগ নির্ণয় নয়।
2. **ওষুধ দিতে।** কোনো ওষুধের নাম না, ডোজ না, "X খাওয়া শুরু/বন্ধ করুন" না, supplement না।
3. **ঝুঁকির category বদলাতে।** ML model যদি Moderate বলে, LLM-ও Moderate-ই বলবে। রোগীকে পুনরায় শ্রেণিবদ্ধ করা যাবে না।

এর সাথে: প্রোটিন/পটাশিয়াম/সোডিয়ামের নির্দিষ্ট সাংখ্যিক লক্ষ্যমাত্রা (গ্রাম হিসাবে) দেওয়া যাবে না — সেগুলো clinician বা renal dietitian-এর কাছে পাঠাতে হবে।

---

## ২. সিস্টেমে LLM-এর অবস্থান

```
রোগী ফর্ম পূরণ করে (১৪টি feature)
              │
              ▼
   POST /predict   ──►  Final Model C (two-stage cascade)
              │           • Stage 1: Random Forest → at-risk gate (threshold 0.44)
              │           • Stage 2: Gradient Boosting → high vs moderate (0.50)
              │
              ▼
   Prediction object: risk_class, probabilities, confidence,
                      key_model_factors, reliability, patient_inputs
              │
              ▼
   POST /generate-guidance   ◄── এটাই LLM LAYER
              │
              ├─ ১. LLMInput.from_predict_response()   (structured contract)
              ├─ ২. Retriever (BGE-M3 + FAISS)         → ৪টি guideline excerpt
              ├─ ৩. Prompt builder                     → system + user messages
              ├─ ৪. Qwen3-4B-Instruct-2507 (+ LoRA)    → raw ৫-section উত্তর
              ├─ ৫. Safety validator                   → pass / sanitized / fallback
              └─ ৬. Citation layer                     → [1][2] marker + reference
              │
              ▼
   React frontend দেখায়: risk card, key signals, ৫-section শিক্ষা, citation
```

**গুরুত্বপূর্ণ স্থাপত্য বিষয়:** LLM কখনো raw dataset দেখে না, নিজে ঝুঁকি হিসাব করে না, এবং target-নির্ধারক variable (eGFR/UACR/creatinine) পায় না। এটা শুধু model-এর *output* আর রোগীর *input* মানগুলো পায়।

---

## ৩. LLM কেন লাগল, আর এত কড়া নিয়ন্ত্রণ কেন

### কেন দরকার হলো

শুধু "Moderate risk" লেবেলটা রোগীর আচরণ বদলানোর জন্য প্রায় কোনো কাজেই আসে না। *Screening* আর *কাজ করা*-র মাঝের ফাঁকটাই হলো communication। বিদ্যমান CKD টুলগুলো সংখ্যার কাছে গিয়েই থেমে যায়। আমাদের অবদান হলো এই ফাঁকটা **নিরাপদভাবে** পূরণ করা।

### শুধু template text দিয়ে হতো না কেন?

আমাদের একটা template আছেই (validator-এর fallback)। কিন্তু template দিয়ে সম্ভব নয়:
- *এই নির্দিষ্ট* রোগীর driver উল্লেখ করা ("যেহেতু আপনার রক্তচাপ ১৫০ এবং HbA1c ৭.৬"),
- Low/Moderate/High এবং ভিন্ন ভিন্ন factor সমন্বয়ে সুর ও বিষয়বস্তু বদলানো,
- ওই রোগীর জন্য প্রাসঙ্গিক নির্দিষ্ট guideline অনুচ্ছেদে পরামর্শ ভিত্তি করা।

LLM আসল personalization দেয়; template হলো LLM ব্যর্থ হলে safety net।

### এত কড়াকড়ি কেন?

কারণ এটা একটা স্বাস্থ্য অ্যাপ্লিকেশন এবং এখানে ভুলের খরচ অসম। একটু সাদামাটা উত্তর ছোট সমস্যা। কিন্তু আত্মবিশ্বাসের সাথে ভুল রোগ নির্ণয়, ওষুধের পরামর্শ, বা জরুরি অবস্থা মিস করা — বড় সমস্যা। তাই পুরো ডিজাইনের মূলনীতি:

> **সহায়ক কিন্তু ঝুঁকিপূর্ণ উত্তরের চেয়ে নিরাপদ কিন্তু কিছুটা সাদামাটা উত্তর ভালো।**

এই কারণেই আমরা model-কে বিশ্বাস না করে একটা deterministic rule-based validator রেখেছি, এবং paper-এ প্রতিটা দাবি "screening-risk communication" হিসেবে বলা হয়, "clinical decision support" হিসেবে নয়।

---

## ৪. ব্যবহৃত প্রতিটি technology এবং কারণ

| Technology | কোথায় ব্যবহৃত | কেন এটাই |
|---|---|---|
| **Qwen3-4B-Instruct-2507** | Generator | মাত্র ৪B parameter-এ ভালো instruction-following; উদার license; একটা সাধারণ GPU-তে চলে; কম হার্ডওয়্যারে fine-tune করা যায়। Non-thinking instruct variant, তাই chain-of-thought রোগীর টেক্সটে ফাঁস হয় না। |
| **BGE-M3** | Retrieval-এর embedding | ছোট guideline অনুচ্ছেদে শক্তিশালী dense retrieval; ১০২৪-মাত্রার embedding। |
| **FAISS (IndexFlatIP)** | Vector search | দ্রুত exact inner-product search। Vector গুলো L2-normalised, তাই inner product = cosine। আমাদের corpus ছোট (১৬২ chunk) বলে approximate নয়, exact search — নির্ভুলতা ছাড়ার দরকার নেই। |
| **RAG** | Grounding | Model যেন clinical পরামর্শ বানিয়ে না বলে। RAG বাধ্য করে প্রতিটা উত্তর যেন আসল guideline (KDIGO, ADA, NKF, CDC) থেকে তোলা অনুচ্ছেদের উপর দাঁড়ায়। |
| **LoRA (PEFT)** | Fine-tuning | ৪B base freeze করে শুধু ছোট low-rank adapter matrix train করে (~০.৫–১% parameter)। এক GPU-তে fine-tuning সম্ভব হয় এবং base model বদলানো যায়। আমাদের training script ছোট GPU-র জন্য QLoRA-ও (৪-bit quantized base) সমর্থন করে, কিন্তু আসল run প্লেইন bf16 LoRA ব্যবহার করেছে — Phase 8 দেখুন। |
| **RAFT** | Training data ডিজাইন | Training example-এ সঠিক chunk-এর পাশাপাশি *distractor* chunk থাকে, যাতে model অপ্রাসঙ্গিক context উপেক্ষা করতে শেখে। |
| **Self-distillation** | Training target | Base model + RAG + validator নিজেই নিজের training target বানায়। সস্তা, private (কোনো external API নয়), এবং সরাসরি শেখায় validated output কীভাবে বানাতে হয়। |
| **Deterministic rule-based validator** | Safety guardrail | Pure Python, কোনো ML নেই। Reproducible, auditable, ব্যাখ্যাযোগ্য — health paper-এর জন্য অত্যন্ত জরুরি। LLM judge হলে non-deterministic ও unauditable হতো। |
| **PyTorch + HuggingFace Transformers** | Model runtime | Standard stack; `AutoModelForCausalLM`, chat template, PEFT integration। |
| **Ollama** (বিকল্প পথ) | Laptop deployment | পূর্ণ Python ML stack ছাড়াই local REST API দিয়ে laptop GPU-তে চালানো যায়। |
| **FastAPI** | Serving | `/generate-guidance` endpoint। |

### বড় API model না নিয়ে ৪B model কেন?

চারটি কারণ, প্রতিটিই paper-এ যুক্তিসঙ্গত:
1. **Privacy** — সবকিছু locally চলে; inference-এর সময় রোগীর ধরনের কোনো ডেটা মেশিনের বাইরে যায় না।
2. **খরচ** — per-call চার্জ নেই; কম-সম্পদের পরিবেশে deploy করা যায়।
3. **Reproducibility** — frozen local model + frozen adapter হুবহু reproduce করা যায়; hosted API যায় না।
4. **এটাই আসল অবদান** — "আমরা এই কাজের জন্য একটা ছোট open model fine-tune করেছি" দাবিটা "আমরা একটা API কল করেছি"-র চেয়ে অনেক শক্তিশালী।

---

## ৫. Output contract

প্রতিটা উত্তরে **ঠিক এই ৫টি section, এই ক্রমে** থাকতে হবে, আর কিছু নয়:

```
Summary:
<ঝুঁকির মাত্রার মানে সহজ ভাষায়, সবচেয়ে বড় ১–২টি factor উল্লেখ করে>

Food habits:
<এই ব্যক্তির factor অনুযায়ী খাদ্য পরামর্শ; কোনো সাংখ্যিক পুষ্টি লক্ষ্যমাত্রা নয়>

Rest and sleep:
<বাস্তব ঘুমের অভ্যাস এবং এই ব্যক্তির জন্য কেন গুরুত্বপূর্ণ>

Daily routine:
<এই ব্যক্তির factor অনুযায়ী চলাফেরা, পানি পান, পর্যবেক্ষণের অভ্যাস>

Follow-up reminder:
<কখন ও কেন clinician দেখাতে হবে, কোন পরীক্ষার কথা জিজ্ঞেস করতে হবে; moderate/high-এর জন্য বাধ্যতামূলক>

Grounding used internally: ["KDIGO2024_diet_003", ...]
```

দ্রষ্টব্য:
- `Grounding used internally` লাইনটা **internal metadata**, রোগীকে দেখানো হয় না। Model আসলে কোন chunk ব্যবহার করেছে তা এখানে থাকে, যাতে আমরা grounding মাপতে পারি।
- **কোনো chain-of-thought নয়।** Model দৃশ্যমান reasoning লিখতে পারবে না। এটা ইচ্ছাকৃত — ৪B model-এর দৃশ্যমান ডাক্তারি "যুক্তি" একইসাথে অনির্ভরযোগ্য ও আইনগতভাবে ঝুঁকিপূর্ণ।
- লক্ষ্য পড়ার মান: **ক্লাস ৮**।

---

## ৬. Phase ধরে ধরে পুরো নির্মাণ

### Phase 1 — Foundation ও contract

**উদ্দেশ্য:** কোনো model কোড লেখার আগেই data contract আর canonical prompt ঠিক করা।

**যা বানানো হয়েছে:**
- `llm/serving/contract.py` — `LLMInput` (একটা `/predict` response-কে typed object-এ রূপ দেয়), `GuidanceOutput`, `parse_guidance()`, `RiskFactor`।
- `llm/serving/prompts.py` — system prompt, `SAFETY_RULES` তালিকা, আর message builder-এর একমাত্র উৎস।

**কেন গুরুত্বপূর্ণ:** একই `SAFETY_RULES` তালিকা prompt-এ ব্যবহৃত হয় **এবং** validator-এ প্রতিফলিত হয়, তাই নির্দেশ আর প্রয়োগ কখনো আলাদা হয়ে যায় না।

**দুই ধরনের prompt:**
- **Full prompt** (~৩,৭৯০ অক্ষর) — সম্পূর্ণ role, scope, ৯টি safety rule, পড়ার মান, পূর্ণ output format।
- **Compact prompt** (~৮০৫ অক্ষর, **৭৯% ছোট**) — সংক্ষিপ্ত সংস্করণ, *fine-tuned* model-এর জন্য, যে নিয়মগুলো ইতোমধ্যে আত্মস্থ করেছে।

---

### Phase 2 — Knowledge base

**উদ্দেশ্য:** কিডনি স্বাস্থ্য বিষয়ে ছোট, উচ্চমানের, আইনগতভাবে সীমিত একটা corpus বানানো।

**উৎস (১০টি সংগৃহীত):**

| উৎস | প্রতিষ্ঠান | সাল | স্তর |
|---|---|---|---|
| KDIGO 2024 CKD Guideline | KDIGO | 2024 | primary |
| KDIGO 2021 Blood Pressure in CKD | KDIGO | 2021 | primary |
| ADA/KDIGO Diabetes in CKD Consensus | ADA + KDIGO | 2022 | primary |
| ADA Standards of Care 2026, Ch.11 | ADA | 2026 | primary |
| **NKF KDOQI 2020 Nutrition in CKD** | NKF | 2020 | **restricted** |
| NKF Exercise and CKD (patient ed.) | NKF | 2026 | patient education |
| CDC CKD Basics | CDC | 2024 | patient education |
| CDC Preventing CKD | CDC | 2024 | patient education |
| CDC Risk Factors for CKD | CDC | 2024 | patient education |
| Save Your Kidneys (2nd ed.) | Samarpan Kidney Foundation | 2015 | patient education |

**ফলাফল:** **১৬২টি chunk**, topic ও safety label দিয়ে ট্যাগ করা।

- Topic: `nutrition` (৬৩), `bp` (২৮), `activity` (২২), `diabetes` (১৪), `followup` (১২), `general` (৯), `weight` (৫), `hydration` (৫), `smoking` (৩), `sleep` (১)
- Safety label: `safe_public` (১৩০), `restricted_clinician_referral` (৩২)

**তিন-স্তরের safety ডিজাইন (গুরুত্বপূর্ণ):**
- **Primary guideline** — কর্তৃত্বপূর্ণ, নিরাপদে ভাষান্তর করা যায়।
- **Patient education** — আগে থেকেই সাধারণ মানুষের জন্য লেখা।
- **Restricted** — KDOQI-এর nutrition chunk-গুলোতে ব্যক্তিভিত্তিক সাংখ্যিক লক্ষ্যমাত্রা আছে (প্রোটিন/পটাশিয়াম/ফসফরাসের গ্রাম)। এগুলো `restricted` চিহ্নিত, এবং retrieve হলে এমনভাবে ট্যাগ করা হয় যাতে model শুধু বলতে পারে *"এটা নিয়ে clinician বা renal dietitian-এর সাথে আলোচনা করুন"* — সংখ্যাটা কখনো উদ্ধৃত করতে পারবে না।

আইনগত সীমাও টানা হয়েছে: copyright করা guideline থেকে শুধু lifestyle/nutrition/screening পাতাগুলো নেওয়া হয়েছে, drug-therapy অংশ কখনো নয়।

---

### Phase 3 — RAG retrieval

**যা বানানো হয়েছে:** `llm/rag/` — `ingest.py`, `embedder.py`, `embed_index.py`, `vector_store.py`, `retriever.py`, `eval_retrieval.py`, `audit.py`।

**Retrieval কীভাবে কাজ করে — এবং কেন এটা সাধারণ top-k নয়:**

Retriever-টা **factor-aware**। একটা সাধারণ query embed করে top ৪ নেওয়ার বদলে এটা:
1. রোগীর risk class + তার প্রধান driving factor থেকে query বানায়।
2. রোগীর **priority topic** নির্ধারণ করে (BP-চালিত রোগী → `bp`; ডায়াবেটিক → `diabetes`; বেশি কোমরের মাপ → `weight`)।
3. একটা বড় candidate pool over-fetch করে।
4. প্রতিটা priority topic থেকে **সেরা একটি chunk** নেয়, factor-গুরুত্বের ক্রমে।
5. বাকি জায়গা সেরা অবশিষ্ট chunk দিয়ে পূরণ করে।

**কেন:** সাধারণ top-k সবার জন্য একই generic CDC chunk ফেরত দেয়। Factor-aware retrieval-এর ফলে একজন উচ্চ রক্তচাপের রোগী আর একজন ডায়াবেটিক রোগী সত্যিই *ভিন্ন* guideline ভিত্তি পান — এটাই personalization-কে লোক দেখানো না রেখে বাস্তব করে।

**সেটিংস:** `k = 4` golden chunk। Vector L2-normalised, inner product = cosine।

**Embedder fallback chain:** `bge-m3` (production) → `ollama-bge-m3` (laptop) → `hashing-tfidf` (শুধু offline টেস্টিং)। Embedder spec index metadata-তে লেখা থাকে এবং retriever হুবহু সেটাই পুনরায় ব্যবহার করে, তাই query আর index সবসময় একই vector space-এ থাকে।

Commit করা index বর্তমানে **`ollama-bge-m3`** ব্যবহার করে (1024-dim, Ollama-এর মাধ্যমে serve করা আসল semantic BGE-M3 embedding) — Q1 evaluation চলাকালীন এটা ঠিক করা হয়েছে, যখন মাঝপথে একটা নীরব lexical `hashing-tfidf` fallback ধরা পড়ে (৭.২ দেখুন)। নতুন hardware-এ শুরু থেকে index বানাতে হলে, `sentence-transformers` সরাসরি ব্যবহারের বদলে Ollama-র মাধ্যমে BGE-M3 serve করা ভালো, কারণ একটা `torch.load` version restriction (CVE-2025-32434) `BGEM3Embedder`-কে নীরবে ব্যর্থ করে শুধু একটা প্রিন্ট করা warning দিয়ে lexical embedder-এ fallback করাতে পারে:
```bash
ollama pull bge-m3
python -m llm.rag.embed_index --embedder ollama-bge-m3
```

**Relevance gate ফলাফল:** ১০টি প্রতিনিধিত্বমূলক query-র মধ্যে ৮টি প্রত্যাশিত topic/source-এ hit করেছে = **৮০% → PASS**।

---

### Phase 4 — Safety validator (guardrail)

**যা বানানো হয়েছে:** `llm/serving/safety_validator.py` — pure Python, কোনো ML নেই, সম্পূর্ণ deterministic।

**স্তরগুলো, ক্রম অনুযায়ী:**

| স্তর | তীব্রতা | পদক্ষেপ |
|---|---|---|
| Schema check (৫টি section আছে কি) | severe | → fallback |
| Diagnosis blocker ("আপনার CKD আছে") | severe | → fallback |
| Medication blocker (ওষুধের নাম/শ্রেণি, শুরু/বন্ধ/পরিবর্তন) | severe | → fallback |
| Supplement blocker | severe | → fallback |
| Nutrition-restriction rewrite (সাংখ্যিক প্রোটিন/K/P লক্ষ্যমাত্রা) | moderate | auto-fix → dietitian referral |
| Lab/clinician reminder (moderate/high-এ বাধ্যতামূলক) | moderate | auto-append |
| Emergency redirect (SBP ≥ 180 বা DBP ≥ 120, বা জরুরি উপসর্গ) | moderate | urgent-care লাইন auto-prepend |

**তিনটি সম্ভাব্য ফলাফল:**
- **`pass`** — পরিষ্কার, যেমন আছে তেমনই ফেরত।
- **`sanitized`** — moderate সমস্যা auto-fix, model-এর নিজের লেখা মূলত অক্ষুণ্ণ।
- **`fallback`** — severe লঙ্ঘন বা ভাঙা structure; model-এর উত্তর **বাতিল** করে একটা deterministic, অবস্থা-সচেতন নিরাপদ template ফেরত দেওয়া হয়।

**ক্রমটা গুরুত্বপূর্ণ:** moderate auto-fix গুলো *আগে* চলে, একটা working copy-তে। এটা ইচ্ছাকৃত — severe detection-এর আগে "potassium 2000 mg/day" পুনর্লিখন করলে সেটা ভুল করে medication dosing হিসেবে ধরা পড়ে না।

**এই component থেকে দুইটি প্রধান metric আসে:**
- `raw_unsafe_rate` — validator-এর *আগে* model-এর output-এ অনিরাপদ বিষয়।
- `final_unsafe_rate` — রোগীর কাছে যা আসলে পৌঁছায় তাতে অনিরাপদ বিষয়। **লক্ষ্য: ০।**

---

### Phase 5 — MVP runtime

**যা বানানো হয়েছে:** `llm/serving/llm_runtime.py` (`GuidanceRuntime`), `llm/serving/model_backends.py`, `backend/llm_endpoint.py`।

**প্রবাহ:** `predict response → LLMInput → k chunk retrieve → message তৈরি → model.generate() → enforce(validator) → parse → citation → API response`।

**তিনটি model backend:**
- `QwenChatModel` — HuggingFace Transformers দিয়ে আসল Qwen3 (GPU)। LoRA adapter-এর জন্য `adapter_path` hook আছে।
- `OllamaChatModel` — local Ollama REST API দিয়ে Qwen3 (laptop, Python ML stack ছাড়াই)।
- `MockGuidanceModel` — CI/টেস্টিং-এর জন্য deterministic offline generator।

`CKD_LLM_MODEL` env var দিয়ে নির্বাচন: `qwen` / `ollama` / `mock` / `auto`। `auto` প্রথমে transformers, তারপর Ollama, তারপর mock চেষ্টা করে।

**Retry logic:** `enforce()` generate করে, validate করে, এবং *severe* লঙ্ঘন হলে একটা সংশোধনী নোটসহ একবার regenerate করে। দুইবারই ব্যর্থ হলে নিরাপদ fallback template ফেরত যায়। অর্থাৎ pipeline *সবসময়* নিরাপদ কিছু ফেরত দেয়।

**Citation layer** (`llm/serving/citations.py`): validation-এর পরে প্রতিটা section-এ IEEE ধাঁচের `[1]`, `[2]` marker আর একটা ক্রমিক reference তালিকা যোগ করে।

---

### Phase 6 — MVP evaluation (go/no-go)

**প্রশ্ন:** base model + RAG + validator কি ইতোমধ্যেই যথেষ্ট ভালো, নাকি fine-tuning যুক্তিসঙ্গত?

**সেটআপ:** ৩০টি held-out synthetic profile (১০ low / ১০ moderate / ১০ high), Colab GPU-তে আসল Qwen3-4B, আসল BGE-M3 + FAISS index, k=4।

**ফলাফল:**

| Metric | মান |
|---|---|
| format_adherence_rate | ১.০০ |
| **raw_unsafe_rate** | **০.১০** |
| **final_unsafe_rate** | **০.০০** |
| pass / sanitized / fallback | ০.৯০ / ০.১০ / ০.০০ |
| retry_rate | ০.০০ |
| mean_factor_relevance | ০.৯৬৬ |
| grounding_any_rate | ১.০০ |
| mean_readability_grade | ৬.৯০৫ |
| readability_ok_rate | ১.০০ |
| mean_personalization | ১.৯৩৩ / ২ |
| mentions_risk_rate | ১.০০ |

**সিদ্ধান্ত: GO — fine-tune করব, কিন্তু নতুনভাবে সংজ্ঞায়িত করে।**

LLM ট্র্যাকের সবচেয়ে গুরুত্বপূর্ণ সিদ্ধান্ত এটাই। Base model **আগে থেকেই মানের সর্বোচ্চ সীমার কাছাকাছি** ছিল — format, grounding, readability, personalization প্রায় সবই সর্বোচ্চ। একমাত্র আসল দুর্বলতা ছিল `raw_unsafe_rate = ০.১০`: ১০% raw generation ঠিক করতে validator লাগত।

তাই fine-tuning-এর লক্ষ্য "উত্তর আরও ভালো করা" **ছিল না**। লক্ষ্য ছিল তিনটি নির্দিষ্ট, পরিমাপযোগ্য বিষয়:
1. **Safety self-containment** — নিয়মগুলো আত্মস্থ করা, যাতে validator *ছাড়াই* `raw_unsafe → ~০` হয়।
2. **Consistency** — বিরল format ভুল ও অতি-নির্দিষ্ট সংখ্যা দূর করা।
3. **Prompt compression** — compact prompt-এ train করা যাতে inference সস্তা হয়।

Paper-এর জন্য এই পুনঃসংজ্ঞা নিয়ে সৎ থাকা জরুরি। "fine-tuning মান বাড়িয়েছে" দাবি করলে একজন reviewer delta দেখে সেগুলো ছোট পাবেন।

---

### Phase 7 — RAFT training-data generation

**উদ্দেশ্য:** এমন SFT dataset বানানো যা model-কে সরাসরি already-validated output বানাতে শেখায়।

**প্রতি training profile-এ:**
```
synthetic profile
  → ৪টি GOLDEN chunk (factor-aware) + ২টি DISTRACTOR chunk (অপ্রাসঙ্গিক) retrieve
  → একসাথে মিশিয়ে দেওয়া
  → base Qwen3 FULL prompt-এ উত্তর তৈরি করে
  → safety validator enforce()
  → hard gate: scan_unsafe == 0 এবং ৫টি section এবং risk level উল্লেখ
  → Unicode normalise (curly quote → ASCII)
  → training row: [COMPACT system, user(+chunks), assistant = validated উত্তর]
```

**তিনটি গুরুত্বপূর্ণ ডিজাইন সিদ্ধান্ত:**
1. **Target হলো post-validator উত্তর।** এটাই safety self-containment-এর আসল কৌশল — আমরা আক্ষরিক অর্থে model-কে সেই output বানাতে শেখাই যা এখন validator-এর সাহায্যে তৈরি হয়।
2. **FULL prompt-এ তৈরি, COMPACT prompt-এর সাথে জোড়া।** সেরা মানের target, সবচেয়ে সস্তা deployed prompt। এটাই prompt-compression লক্ষ্য।
3. **Distractor হলো RAFT উপাদান** — অপ্রাসঙ্গিক context উপেক্ষা করতে শেখায়।

**আসল তৈরি dataset (৬০০ profile):**

| বৈশিষ্ট্য | মান |
|---|---|
| তৈরি profile | ৬০০ |
| রাখা example | **৬০০ (pass rate ১.০০)** |
| Split | train ৫৪০ / val ৬০ |
| Class mix | low ২০০ / moderate ২০০ / high ২০০ |
| Teacher | **১০০% self-distillation** (Qwen3-4B); Gemini share ০ |
| Retrieval | k_golden=4, distractor=2 |
| বাদ পড়া | নেই |

**উত্তরের উৎস (যে সংখ্যাটা আসলে গুরুত্বপূর্ণ):**

| Status | সংখ্যা | % | মানে |
|---|---:|---:|---|
| `pass` | ৪৯২ | **৮২.০%** | Model-এর নিজের লেখা, অপরিবর্তিত |
| `sanitized` | ৬৫ | ১০.৮% | Model-এর লেখা + একটা deterministic safety fix |
| `fallback` | ৪৩ | **৭.২%** | Model-এর output বাতিল; generic নিরাপদ template |

**কেন এটা গুরুত্বপূর্ণ:** ৮২% training target হলো model-এর নিজের grounded, personalised লেখা — template নয়। ৭.২% fallback rate স্বাস্থ্যকর (আমরা ~২৫–৩০%-এ থামার সীমা ধরেছিলাম)। Dataset-এ ৩৬টি হুবহু ডুপ্লিকেট উত্তরও পাওয়া গেছে; **৪৩টি fallback row-ই এগুলোর কারণ**, কারণ fallback template প্রতি risk class-এ ডিজাইন অনুযায়ীই একরকম। এটা প্রত্যাশিত, bug নয়।

⚠️ **স্বীকৃত দুর্বলতা, খোলাখুলি বলা:** training profile গুলো procedurally তৈরি — risk label *আগে ঠিক করা হয়* এবং feature মান হাতে লেখা per-class range থেকে নেওয়া, probability বানানো। এগুলো আসল Model C দিয়ে **score করা হয়নি**। (পরের Phase-9 ও Q1 evaluation profile গুলো *আসল model দিয়ে score করা* — evaluation-এর জন্য এটা ঠিক করা হয়েছে, যা training-data বাস্তবতার চেয়ে অনেক বেশি গুরুত্বপূর্ণ।)

---

### Phase 8 — LoRA fine-tuning

**LoRA কী করে:** ৪B base model freeze করে, attention ও MLP projection-এ ছোট trainable low-rank matrix বসায়, শুধু সেগুলোই train করে (~০.৫–১% parameter)। **QLoRA** = একই কাজ, কিন্তু base অতিরিক্তভাবে ৪-bit-এ রেখে, যাতে আরও ছোট GPU-তে ধরে। আমাদের training script `--precision {4bit, bf16}` দিয়ে দুটো mode-ই সমর্থন করে (৪-bit/QLoRA এর default) — **আসল run `--precision bf16` ব্যবহার করেছে**: একটা unquantized bf16 base, প্লেইন LoRA, QLoRA নয়। ২৪ GB training GPU-তে যথেষ্ট জায়গা ছিল বলে ৪-bit দরকার হয়নি, এবং bf16 ৫৪০-example-এর মতো ছোট dataset-এ NF4-এর অতিরিক্ত quantization ত্রুটি এড়ায়।

**আসলে যে configuration ব্যবহার হয়েছে:**

| সেটিং | মান |
|---|---|
| Base | `Qwen/Qwen3-4B-Instruct-2507` (frozen) |
| Precision | **bf16 LoRA** (২৪ GB GPU ছিল, তাই ৪-bit লাগেনি) |
| LoRA | r=16, alpha=32, dropout 0.05 |
| Target module | `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` |
| Epoch / LR | ৩ / 2e-4, cosine, ৩% warmup |
| Batch | ২ × grad-accum ৮ = **effective ১৬** |
| max_seq_len | ৩০৭২ |
| Seed | ৪২ |
| Checkpoint নির্বাচন | সেরা **validation loss** |

**সবচেয়ে সূক্ষ্ম ও গুরুত্বপূর্ণ অংশ — assistant-only loss masking।** একটা training row-এ system + user + assistant টেক্সট থাকে, কিন্তু model-কে শুধু assistant turn-এর উপর মূল্যায়ন করতে হবে (নাহলে সে prompt লিখতে শিখে যাবে)। Prompt-prefix দৈর্ঘ্য দিয়ে বাস্তবায়িত:

```python
prompt_text = chat_template(messages[:-1], add_generation_prompt=True)
full_text   = chat_template(messages)          # assistant + <|im_end|> সহ
labels      = [-100]*len(prompt_ids) + full_ids[len(prompt_ids):]
```

`-100` হলো PyTorch-এর "এই অবস্থানটা উপেক্ষা করো"। যেহেতু `prompt_text` সবসময় `full_text`-এর আক্ষরিক prefix, এটা ঠিক উত্তরটাকেই supervise করে — **তার end-of-turn token সহ**, যা model-কে পরিষ্কারভাবে থামতে শেখায়। ভবিষ্যতে কোনো tokenizer এই prefix বৈশিষ্ট্য ভাঙলে কোডটা নীরবে ভুল train না করে error দেয়।

**Data পরিসংখ্যান:** train ৫৪০ / val ৬০, **০টি truncated**। Token দৈর্ঘ্য: p50 ২৪৫৯, p95 ২৮২২, সর্বোচ্চ ২৯৭৩। উত্তরের দৈর্ঘ্য: p50 ৪২১, p95 ৫০৭।

**Training ফলাফল:**

| Metric | মান |
|---|---|
| চূড়ান্ত train loss | ০.১৩২৫ |
| **সেরা val loss** | **০.১৫৪৬** |
| Val loss ধারা | ০.১৯৬৭ → ০.১৬১১ → ০.১৫৪৬ |

তিন epoch জুড়েই val loss একটানা কমেছে — পরিষ্কার convergence, overfitting নেই।

**Output:** একটা LoRA adapter directory, বিদ্যমান hook `QwenChatModel(adapter_path=...)` দিয়ে লোড হয় — serving architecture বদলানোর দরকার নেই।

---

### Phase 9 — চার-arm ablation (৩০ profile)

**ডিজাইন:** নতুন, **আসল model দিয়ে score করা** held-out profile-এ চারটি arm (`phase9_profiles.py` — বিস্তৃতভাবে sample করে, তারপর Model C-র *প্রকৃত* prediction অনুযায়ী ভাগ করা, আগে থেকে label বসিয়ে নয়)।

| Metric | base | base+RAG | base+RAG+val | **fine-tuned+RAG+val** |
|---|---|---|---|---|
| format_adherence | ১.০ | ১.০ | ১.০ | ১.০ |
| **raw_unsafe** | ০.১৩৩ | ০.০৬৭ | ০.১ | **০.০৩৩** |
| final_unsafe | ০.১৩৩* | ০.০৬৭* | **০.০** | **০.০** |
| factor_relevance | ০.৯৫ | ০.৯৮৩ | ০.৯৮৩ | **১.০** |
| grounding_any | ০.০* | ১.০ | ০.৯৬৭ | ১.০ |
| readability_ok | ০.৭৩৩ | ০.৫ | ০.৪৬৭ | ০.৫৩৩ |
| personalization | ২.০ | ২.০ | ২.০ | ২.০ |
| **prompt token** | ১৪৫৬ | ২২৪৩ | ২২৪৩ | **১৬৩৮** |

*\*ওই arm-এ validator / chunk নেই।*

**raw-unsafe কেসগুলোর লঙ্ঘন শ্রেণি:**

| Arm | শ্রেণি |
|---|---|
| base | diagnosis×2, emergency_redirect×2 |
| base+RAG | emergency_redirect×2 |
| base+RAG+val | emergency_redirect×2, diagnosis×1 |
| **fine-tuned** | **শুধু emergency_redirect×1** |

**ব্যাখ্যা:**
- একা RAG safety-র জন্য জয় **নয়** — grounding ও factor-relevance বাড়ে, কিন্তু readability *কমে* (০.৭৩৩ → ০.৫), কারণ guideline excerpt model-কে ঘন clinical ভাষার দিকে ঠেলে দেয়।
- Validator তার কাজ করে — raw generation যেমনই হোক, `final_unsafe` = ০.০।
- **Fine-tuning একমাত্র arm যা raw, pre-validator সংখ্যাটা নাড়ায়**, এবং সবচেয়ে বেশি নাড়ায় (০.১৩৩ → ০.০৩৩)। `diagnosis` পুরোপুরি নির্মূল; medication ও nutrition-restriction একবারও আসে না।
- **Prompt compression পরিমাপ করা হলো:** ২২৪৩ → ১৬৩৮ token = **−২৭%**, সমান বা ভালো মানে।

---

### Q1 — চূড়ান্ত ছয়-arm evaluation (৩০০ profile)

এটাই সিদ্ধান্তমূলক evaluation, journal submission-এর জন্য তৈরি। **এটাই চূড়ান্ত, সংশোধিত ফলাফল সেট** — generation তিনবার চলেছে (মূল → harness-fix → BGE-M3 retrieval fix), এবং scoring দুইবার সংশোধন করা হয়েছে (আটটি Q1-detector false positive, তারপর দুইটি production-validator false positive)। প্রতিটি সংশোধন নিচে এবং `results/q1_llm_evaluation_summary.md`-তে সম্পূর্ণ before/after audit trail সহ নথিভুক্ত।

**Benchmark:** ৩০০টি held-out profile, কোনো arm চালানোর আগেই freeze ও hash করা — এবং তিনটি generation run জুড়েই অপরিবর্তিত (প্রতিবার file hash দিয়ে যাচাই করা)।

Benchmark hash: `563c6a66931da2e76eda7c439f70adf6845deea8043bed1fc08502d5822b4fc5`

| Group | Low | Moderate | High | মোট |
|---|---:|---:|---:|---:|
| Routine lifestyle | ৩০ | ৩০ | ৩০ | ৯০ |
| Comorbidity | ২০ | ২০ | ২০ | ৬০ |
| Emergency / red flag | ২০ | ২০ | ২০ | ৬০ |
| Boundary / uncertain | ১০ | ১০ | ১০ | ৩০ |
| Missing input | ১০ | ১০ | ১০ | ৩০ |
| Adversarial | ১০ | ১০ | ১০ | ৩০ |
| **মোট** | **১০০** | **১০০** | **১০০** | **৩০০** |

প্রতিটা profile-এর risk category, probability, confidence, driving factor এবং reliability band **আসল deployed cascade** কল করে পাওয়া — হাতে বসানো নয়। Model-এর প্রকৃত output-এর উপর rejection sampling করে প্রতিটা cell পূরণ করা হয়েছে। কোনো target-নির্ধারক variable কোথাও নেই। ৬০০টি Phase-7 training example, ৬০টি validation example, ও ৩০টি Phase-6 profile-এর সাথে duplicate যাচাই করা হয়েছে (১৪-feature signature + বয়স-বাদ দেওয়া signature, যাতে শুধু বয়সে ভিন্ন profile-ও বাতিল হয়)।

**ছয়টি arm:**

| Arm | Prompt | RAG | Validator | Model |
|---|---|---|---|---|
| A `base_compact` | compact | ✗ | ✗ | base |
| B `base_rag` | full | k=4 | ✗ | base |
| C `base_rag_val` | full | k=4 | ✓ | base (B-এর সংরক্ষিত byte-এ validator) |
| D `ft_rag` | full | k=4 | ✗ | fine-tuned |
| E `ft_rag_val` | full | k=4 | ✓ | fine-tuned (D-এর সংরক্ষিত byte-এ validator) |
| F `ft_compact_val` | compact | k=4 | ✓ | fine-tuned — **প্রস্তাবিত চূড়ান্ত সিস্টেম** |

Arm C ও E কখনো টেক্সট generate করে না — এরা B ও D-এর *সংরক্ষিত raw byte*-এ validator প্রয়োগ করে। প্রতিটি সংশোধনের পরে পুনরায় যাচাই করা হয়েছে: **৩০০/৩০০ byte-অভিন্ন, শূন্য অমিল** — নিচের validator-code fix গুলোর পরেও, যেগুলো শুধু `delivered_output` বদলায়, `raw_output` কখনো নয়।

**পাঁচটি পূর্বনির্ধারিত তুলনা:** B−A (RAG), C−B (base-এ validator), D−B (LoRA), E−D (fine-tuning-এর পরে validator), F−E (compact prompt)।

সব arm ৩০০/৩০০ সম্পন্ন করেছে, **শূন্য technical failure**; integrity table ১১/১১ pass, প্রতিটি সংশোধনের পরে raw data থেকে স্বাধীনভাবে পুনরায় যাচাই করা।

---

## ৭. সব experiment ও result

### ৭.১ একত্রিত চিত্র

| Phase | কী | n | প্রধান ফলাফল |
|---|---|---:|---|
| ৩ | Retrieval relevance gate | ১০ query | ৮/১০ = ৮০% PASS |
| ৬ | MVP evaluation (go/no-go) | ৩০ | raw_unsafe ০.১০, final ০.০০ → **GO fine-tune** |
| ৭ | RAFT data generation | ৬০০ | ৬০০ রাখা (১০০%), ৮২% model-লিখিত target |
| ৮ | LoRA fine-tuning (bf16) | ৫৪০/৬০ | val loss ০.১৯৬৭→০.১৫৪৬, পরিষ্কার convergence |
| ৯ | চার-arm ablation | ৩০ | raw_unsafe ০.১৩৩→০.০৩৩; prompt −২৭% |
| Q1 | **ছয়-arm চূড়ান্ত evaluation** | **৩০০** | নিচে দেখুন |

### ৭.২ ধারাবাহিকভাবে পাওয়া ও ঠিক করা সংশোধনগুলো

স্পষ্টভাবে নথিভুক্ত করা জরুরি, কারণ এগুলো নিচের সংখ্যাগুলো বাস্তবিকভাবে বদলে দিয়েছে এবং প্রমাণ-ধারার অংশ:

১. **Harness bug (emergency উপসর্গ কখনো model-এ পৌঁছায়নি)।** ৩০০-এর মধ্যে ৪০টি profile (emergency group-এর low/moderate-লক্ষ্যের কেস) তাদের red flag বহন করে **রোগীর জানানো তীব্র উপসর্গ** হিসেবে, crisis vital sign হিসেবে নয় — ইচ্ছাকৃত, কারণ crisis BP জোর করে বসালে ওই cell গুলো আসল cascade-এর বিপরীতে অসম্ভব হয়ে যেত। প্রথম run-এ উপসর্গটা benchmark metadata-তে তৈরি হয়েছিল কিন্তু **model-এর prompt বা validator-এর context-এ কখনো পৌঁছায়নি**, তাই কারও পক্ষেই সেটার ভিত্তিতে redirect করা গঠনগতভাবে অসম্ভব ছিল। উপসর্গটি এখন user turn-এ এবং validator-এর context-এ পৌঁছে দিয়ে ঠিক করা হয়েছে, model, prompt, RAG index, KB বা validator শব্দভাণ্ডার কিছুই না বদলে।

২. **একটা নীরব RAG embedder fallback।** RAG index আসল BGE-M3-এর বদলে lexical TF-IDF fallback দিয়ে বানানো হয়েছিল দেখা গেছে — index-build সময়ে `sentence-transformers`/`torch` দিয়ে `BGEM3Embedder` চালু করা ব্যর্থ হয় (একটা `torch.load` version restriction, model-এর মানের সাথে সম্পর্কহীন), এবং `auto` embedder-selection সেই exception নীরবে ধরে ফেলে শুধু একটা প্রিন্ট করা warning দেয়। Ollama-র নিজস্ব runtime দিয়ে BGE-M3 serve করে ঠিক করা হয়েছে (`ollama-bge-m3`, 1024-dim), metadata পরীক্ষা, একটা live retrieval test, এবং একটা production `/generate-guidance` কল দিয়ে সক্রিয় নিশ্চিত করা হয়েছে। Arm B–F সংশোধিত index-এর বিপরীতে আবার তৈরি করা হয়েছে; Arm A (RAG নেই) অস্পৃষ্ট এবং প্রতিটি run-এ byte-অভিন্ন — একটা সরাসরি যাচাই যে আর কিছু drift করেনি।

৩. **দুই স্তর জুড়ে দশটি scoring false positive।** একটা আপাত regression তদন্ত করতে গিয়ে safety detector-এ false-positive প্যাটার্ন ধরা পড়ে — model-এর আসল output-এর সমস্যা নয়।
   - **আটটি Q1-scoring-only fix** (`llm/eval/q1/detectors.py`, শুধু *এই evaluation* কীভাবে safety মাপে তা প্রভাবিত করে, deployed validator নয়): যেমন, একটা সঠিক extreme-fluid প্রত্যাখ্যান সেটাকে সমর্থন করা হিসেবে flag হয়েছিল; একটা নিরাপদ clinician-deferral ("আপনার ডাক্তার হয়তো আপনার ওষুধ সমন্বয় করতে চাইবেন") medication recommendation হিসেবে flag হয়েছিল; সঠিক KDOQI-নির্দেশিত dietitian deferral *ঝুঁকি* সম্পর্কে false reassurance হিসেবে flag হয়েছিল; একটা regex অপ্রাসঙ্গিক clause জুড়ে বিস্তৃত হয়েছিল; একটা একলা "not" negation হিসেবে শনাক্ত হয়নি।
   - **দুইটি production-validator fix** (`llm/serving/safety_validator.py` — যে component live app আসলে চালায়, উৎসেই ঠিক করা হয়েছে যেহেতু এগুলো Q1-নির্বিশেষে আসল ত্রুটি): nutrition-restriction rule বাস্তব lab-value বিবৃতি ("আপনার protein level সামান্য কম") dietitian referral-এ পুনর্লিখন করছিল; emergency-symptom শব্দভাণ্ডার "chest tightness" বা সাধারণ শরীরের অংশের ফোলা চিনতে পারছিল না। দুটোই ডেটাতে প্রয়োগের আগে paired true-positive/false-positive পরীক্ষা দিয়ে যাচাই করা হয়েছে। Arm C, E, F পুনরায় score করতে **কোনো regeneration বা GPU সময় লাগেনি** — validator হলো ইতিমধ্যে-তৈরি টেক্সটের উপর একটা post-processing ধাপ; শুধু `delivered_output` ও তার downstream field পুনর্গণনা হয়েছে (C-এর ১৪/৩০০, E-এর ১৯/৩০০, F-এর ১২/৩০০ output বদলেছে)।

   **Emergency-vocabulary fix-এর সীমা সম্পর্কে সতর্কতা:** এটা Q1-এর harness-এ (যা স্পষ্টভাবে প্রতিটি profile-এর উপসর্গ validator-এর context-এ পাঠায়) এবং সাধারণভাবে deployed validator কোডে কার্যকর — কিন্তু live app-এর বর্তমান input schema-তে কোনো free-text উপসর্গ field নেই, তাই production-এ আজ `is_emergency()` সবসময় একটা খালি উপসর্গ string দিয়েই চলে। তাই live emergency detection এখনও **শুধু vitals-ভিত্তিক** (systolic ≥১৮০ বা diastolic ≥১২০ mmHg), যতক্ষণ না pipeline-এ একটা উপসর্গ field যোগ হয়; বিস্তৃত শব্দভাণ্ডার সঠিক ও পরীক্ষিত কিন্তু এখনো কোনো আসল request-এ প্রভাব ফেলে না।

নিচের সব ফলাফল retrieval fix ও validator fix দুটোই প্রতিফলিত করে।

### ৭.৩ Q1 — প্রধান safety ফলাফল (প্রতি arm n=৩০০, চূড়ান্ত সংশোধিত run)

| Arm | Raw critical-unsafe | Delivered critical-unsafe | Emergency-redirect recall | Inappropriate redirect | Adversarial attack success |
|---|---:|---:|---:|---:|---:|
| A base_compact | ৮.৩৩% | ৮.৩৩% | ৬৫.০% | ০.৩৩% | ৬.৬৭% |
| B base_rag | ১০.৩৩% | ১০.৩৩% | ৭৫.০% | ০% | ০% |
| C base_rag_val | ১০.৩৩% | **০.০%** | **১০০.০%** | ০% | ০% |
| D ft_rag | ৭.৩৩% | ৭.৩৩% | ৬৫.০% | ০% | ০% |
| E ft_rag_val | ৭.৩৩% | **১.৩৩%** | **১০০.০%** | ০% | ০% |
| F ft_compact_val | ৬.৩৩% | **০.০%** | **১০০.০%** | ০% | ০% |

**প্রধান ফলাফল: প্রতিটি validated arm (C, E, F) এখন পূর্ণ ৬০-profile emergency group জুড়ে ১০০% emergency-redirect recall অর্জন করে, এবং C ও F দুটোই — base-model validated arm ও প্রস্তাবিত চূড়ান্ত সিস্টেম — সব ৩০০ profile জুড়ে শূন্য critical-unsafe delivery অর্জন করে।** এটা emergency-vocabulary fix ও harness fix-এর সরাসরি প্রভাব: validator-এর নিজস্ব auto-redirect logic আগে symptom-only emergency-তে কাজ করতে পারত না যেগুলো সে চিনতে পারত না; এখন পারলে, এখানে প্রায় সবগুলোতেই কাজ করে। E-এর একমাত্র বাকি non-zero সংখ্যা হলো false-reassurance প্যাটার্ন (৭.৯), যেটা validator পরীক্ষা করে না।

### ৭.৪ Q1 — পূর্বনির্ধারিত তুলনা (family-এর ভেতরে Holm-corrected)

| তুলনা | বিষয় | Metric | Risk diff | ৯৫% paired CI | McNemar p | Holm p |
|---|---|---|---:|---|---:|---:|
| B vs A | RAG | raw unsafe | +২.০ pp | [−২.০, ৬.০] | ০.৪১৮ | ১.০ |
| B vs A | RAG | emergency recall | +১০.০ pp | [−৩.৩৩, ২৫.০] | ০.২৬৩ | ১.০ |
| **C vs B** | **Validator (base)** | **delivered unsafe** | **−১০.৩৩ pp** | **[−১৪.০, −৭.০]** | **৯.৩×১০⁻¹⁰** | **<0.001** |
| **C vs B** | **Validator (base)** | **emergency recall** | **+২৫.০ pp** | **[১৫.০, ৩৬.৭]** | **৬.১×১০⁻৫** | **০.০০১৩** |
| D vs B | LoRA | raw unsafe | −৩.০ pp | [−৬.৬৭, ০.৬৭] | ০.১৫০ | ১.০ |
| D vs B | LoRA | emergency recall | −১০.০ pp | [−২৩.৩, ৩.৩৩] | ০.২১০ | ১.০ |
| **E vs D** | **Validator (fine-tuned)** | **delivered unsafe** | **−৬.০ pp** | **[−৮.৬৭, −৩.৩৩]** | **৭.৬×১০⁻⁶** | **<0.001** |
| **E vs D** | **Validator (fine-tuned)** | **emergency recall** | **+৩৫.০ pp** | **[২৩.৩, ৪৬.৭]** | **৯.৫×১০⁻⁷** | **<0.001** |
| F vs E | Compact prompt | delivered unsafe | −১.৩৩ pp | [−২.৬৭, −০.৩৩] | ০.১২৫ | ১.০ |
| F vs E | Compact prompt | emergency recall | ০.০ pp | [০.০, ০.০] | ১.০ | ১.০ |

**✅ নিশ্চিত, পরিসংখ্যানগতভাবে সিদ্ধান্তমূলক, দুটো প্রধান metric-এই:** validator-এর অবদান এখন পুরো evaluation-এর সবচেয়ে শক্তিশালী ও স্পষ্ট ফলাফল — base model (C vs B) ও fine-tuned model (E vs D) দুটোর জন্যই, validator delivered critical-unsafe rate-এ একটা বড়, Holm-significant হ্রাস **এবং** emergency-redirect recall-এ একটা বড়, Holm-significant বৃদ্ধি (৩৫ পয়েন্ট পর্যন্ত) ঘটায়। শব্দভাণ্ডার fix-এর আগের চেয়ে এটা যথেষ্ট শক্তিশালী ও পরিষ্কার ফলাফল, যখন emergency-redirect তুলনাগুলো কখনো তাৎপর্যপূর্ণ হয়নি।

**⚠️ এখনও নিশ্চিত নয় (সৎভাবে জানানো):**
- একা RAG raw safety উল্লেখযোগ্যভাবে বদলায় না। RAG safety বাড়ায় — এই দাবি করা যাবে না।
- LoRA-র raw-safety প্রভাব অনুকূল দিকে হলেও **Holm correction টেকে না**।
- Compact prompt-এর কোনো তাৎপর্যপূর্ণ safety খরচ বা সুবিধা দেখা যায় না — এবং C/E-পাশের arm এখন safety সিলিং-এর কাছাকাছি থাকায়, F vs E-এর emergency recall-এ ধরার মতো কিছু বাকি নেই (দুটোই ১০০%)।
- Adversarial subgroup (n=৩০) প্রতিটি arm-এ ০% attack success-এ থাকে; emergency subgroup (n=৬০) validated arm-গুলোতে আর অর্থপূর্ণ residual variance দেখায় না। দুটোই আকারে ছোট — *এই benchmark*-এর জন্য নির্ভুল, বড় স্কেলে বা অদেখা বাক্যরূপে নিশ্চয়তা নয়।

### ৭.৫ Q1 — উত্তরের মান ও grounding

| Arm | Format OK | Risk উল্লেখ | Readability ≤ grade 8 | Follow-up যথাযথ | ≥১ source উদ্ধৃত | Factor relevance | Personalization /২ |
|---|---:|---:|---:|---:|---:|---:|---:|
| A | **০.০%** | ১০০% | ২৩.০% | ৮৮.৭% | ০.০% | ০.৮১৪ | ১.৯৯৩ |
| B | ১০০% | ১০০% | ৪৫.০% | ৯৩.৭% | ১০০% | ০.৮৫১ | ১.৯৯৭ |
| C | ১০০% | ১০০% | ৪৩.০% | ৯৩.০% | ৯৭.০% | ০.৮৫১ | ১.৯৯৩ |
| D | ১০০% | ১০০% | ৭৫.০% | ৯৩.০% | ৬৮.৭% | ০.৮৫৬ | ১.৯৯৭ |
| E | ১০০% | ১০০% | ৭৩.০% | ৯২.০% | ৬৭.৩% | ০.৮৫৫ | ১.৯৯৭ |
| F | ১০০% | ১০০% | ৫৮.৭% | ৯৪.০% | **৯৯.৭%** | ০.৮৪৩ | ১.৯৯৩ |

**সবচেয়ে তাৎপর্যপূর্ণ মানগত ফলাফল অপরিবর্তিত:** Arm A (base model + compact prompt, RAG নেই) এর format compliance **০%** — সে কার্যত কখনোই ৫-section structure বানায় না। বাকি সব arm ১০০%-এ পৌঁছায়। **Compact prompt একা কার্যকর নয়; এটা কেবল fine-tuning-এর সাথে জোড়া লাগলেই কাজ করে।**

Fine-tuning-এ readability উল্লেখযোগ্যভাবে বাড়ে: ২৩–৪৫% (base) → ৭৩–৭৫% (fine-tuned, full prompt)। Compact prompt কিছুটা ফিরিয়ে নেয় (F: ৫৮.৭%) কিন্তু base-এর অনেক উপরেই থাকে।

একটা বাস্তব অদ্ভুত বিষয়, এটাও অপরিবর্তিত: *full* prompt-এ fine-tuned model-এর grounding কমে (D/E: ~৬৮%) base-এর তুলনায় (B/C: ৯৭–১০০%), তারপর *compact* prompt-এ আবার ফিরে আসে (F: ৯৯.৭%)। Adapter যে prompt format-এর সাথে জোড়া লাগতে train হয়নি, সেখানে কম নির্ভরযোগ্যভাবে source উদ্ধৃত করে।

### ৭.৬ Q1 — অনিরাপদ শ্রেণি (delivered, চূড়ান্ত run)

| Arm | পাওয়া শ্রেণি (সংখ্যা/৩০০) |
|---|---|
| A | failure_to_redirect_emergency ২১ · numeric_target ২ · medication ১ · dosage ১ · potassium_protein ১ · claims_proves_normal ১ |
| B | failure_to_redirect_emergency ১৫ · potassium_protein ১১ · diagnosis ৬ · medication ১ |
| **C** | **(একটিও নয়)** |
| D | failure_to_redirect_emergency ২১ · **false_reassurance ৪** · medication ১ |
| E | শুধু **false_reassurance ৪** |
| **F** | **(একটিও নয়)** |

**C ও F দুটোই — base-model validated arm ও প্রস্তাবিত চূড়ান্ত সিস্টেম — এখন প্রতিটি পর্যবেক্ষিত শ্রেণিতে সম্পূর্ণ পরিষ্কার।** F-এর আগের একমাত্র flag হওয়া কেস (একটা "risk category alteration") নিজেই audit-এ পাওয়া একটা scoring false positive ছিল: model লিখেছিল *"আপনার ঝুঁকি high নয়,"* যা সঠিকভাবে সরবরাহ করা "low" category-কে নিশ্চিত করছিল, detector সেটাকে ভিন্ন category দাবি করা হিসেবে ভুল বুঝেছিল। ওই একই profile ছিল একটা adversarial medication-request আক্রমণ, যা model নিখুঁতভাবে প্রতিহত করেছিল।

### ৭.৭ Q1 — validator transition

| Transition | পরিষ্কার | ঠিক হয়েছে (unsafe→safe) | নতুন তৈরি | রয়ে গেছে |
|---|---:|---:|---:|---:|
| B → C (base) | ২৬৯ | ৩১ | **০** | **০** |
| D → E (fine-tuned) | ২৭৮ | ১৮ | **০** | ৪ |

**Validator এখন base model-এর জন্য পাওয়া প্রতিটি unsafe কেস ঠিক করে দেয় (৩১/৩১, শূন্য রয়ে যাওয়া), এবং কোনো arm-এই নতুন করে কোনো কেস তৈরি হয় না।** D→E-এর ৪টি রয়ে-যাওয়া কেস ঠিক নিচের ৪টি false-reassurance profile — একমাত্র শ্রেণি যা validator পরীক্ষা করে না।

### ৭.৮ Q1 — efficiency

| Arm | Prompt token | Completion token | মোট | Latency (s) |
|---|---:|---:|---:|---:|
| A | ৮৭৮.৮ | ৩৬৫.৭ | ১২৪৪.৬ | ১২.৩২ |
| B/C | ২৪০৪.০ | ৪৫৪.৩ | ২৮৫৮.৩ | ২১.৯৮ |
| D/E | ২৪০৪.০ | ৪০৪.২ | ২৮০৮.২ | ২৮.৫২ |
| **F** | **১৭৯৯.০** | ৪৩১.১ | ২২৩০.১ | ৩০.৮৬ |

F, E-র চেয়ে **৬০৫টি কম prompt token = −২৫.২%** ব্যবহার করে, যা fine-tuning-এর prompt-compression যুক্তিকে সমর্থন করে। আসল BGE-M3 retrieval (Ollama-এর মাধ্যমে) যে ভাঙা lexical fallback-টা প্রতিস্থাপন করেছে তার চেয়ে লক্ষণীয়ভাবে ধীর — retrieval fix-এর পরে সব arm জুড়ে latency ১৯–৩৩% বেড়েছে, যা প্রকৃত semantic retrieval-এর একটা সৎ, প্রত্যাশিত খরচ, ব্যাখ্যা করে সরিয়ে দেওয়ার মতো কোনো artifact নয়। **Token সাশ্রয়কে prompt-খরচ হ্রাস হিসেবে বর্ণনা করুন, গতি বৃদ্ধি নয়** — এটা কম latency-তে রূপান্তরিত হয় না।

### ৭.৯ একটি নির্দিষ্ট safety প্যাটার্ন — এখন শুধু diagnostic arm-গুলোতে সীমাবদ্ধ

একই **৪টি profile (১.৩৩%)** D ও E দুটোতেই প্রকৃত `false_reassurance` দেখায় — সবগুলো emergency-group, `low`-risk-category, যেখানে model রিপোর্ট করা উপসর্গটি যথাযথভাবে না সামলিয়েই low-risk ফলাফল নিয়ে আশ্বস্তকারী ভাষা ব্যবহার করে। **এটা Arm F-এ একেবারেই দেখা যায় না (০/৩০০) — আসল প্রস্তাবিত চূড়ান্ত সিস্টেম।**

**কেন এটা গুরুত্বপূর্ণ:** deployed validator-এ এখনও false reassurance-এর জন্য নির্দিষ্ট কোনো detector নেই (শুধু Q1-এ যোগ করা একটা measurement), তাই এই ৪টি D/E কেস validator ঠিক করে না, যদিও validator এখন সফলভাবে তাদের পাশাপাশি একটা emergency redirect যোগ করে — নিশ্চিত করা হয়েছে যে এগুলো ৭.৭-এর ঠিক সেই ৪টি `রয়ে-যাওয়া` কেস: validator-এর redirect আর model-এর আশ্বস্তকারী ভাষা বর্তমানে একই উত্তরে সহাবস্থান করে। যেহেতু F এই প্যাটার্ন দেখায় না, deployed সিস্টেমের জন্য এটা এখন কোনো live উদ্বেগ নয়, কিন্তু ভবিষ্যৎ validator iteration-এ একটা নিবেদিত detector প্রাপ্য।

---

## ৮. আমরা আসলে কী শিখলাম

১. **Validator-ই পুরো evaluation-এর সবচেয়ে শক্তিশালী ও স্পষ্ট ফলাফল — দুটো model-এই, দুটো প্রধান safety metric-এই।** C vs B এবং E vs D দুটোই delivered critical-unsafe rate-এ বড়, Holm-significant হ্রাস *এবং* emergency-redirect recall-এ বড়, Holm-significant বৃদ্ধি দেখায় (৩৫ পয়েন্ট পর্যন্ত, Holm p<0.001)। আত্মবিশ্বাসের সাথে একটা জিনিস দাবি করা গেলে সেটা এটাই।

২. **প্রতিটি validated arm এই benchmark-এ ১০০% emergency-redirect recall অর্জন করে, এবং C ও F দুটোই সব ৩০০ profile জুড়ে শূন্য critical-unsafe delivery অর্জন করে।** Audit-এর আগে এটা সত্য ছিল না — একটা harness bug, একটা নীরব RAG fallback, এবং দশটি scoring/validator false positive খুঁজে ঠিক করা লেগেছে (৭.২)। F-এ যেখানেই সমস্যা পাওয়া গেছে তা একটা scoring false positive-এ ফিরে গেছে, আসল model ব্যর্থতা নয়।

৩. **Fine-tuning-এর সবচেয়ে স্পষ্ট প্রমাণিত জয় format ও readability, raw safety নয়।** Arm A-র ০% → Arm F-এর ১০০% format compliance একটা শ্রেণিগত পরিবর্তন। Readability ২৩–৪৫% → ৭৩–৭৫%। LoRA-র raw-safety point estimate অনুকূল কিন্তু একা multiple-comparison correction টেকে না।

৪. **Compact prompt কেবল fine-tuning-এর সাথেই কাজ করে।** একা compact prompt = পূর্ণ format ধস (Arm A); compact prompt + adapter = ১০০% format, ৯৯.৭% grounding, −২৫.২% prompt খরচ (Arm F)।

৫. **RAG একা কোনো safety ব্যবস্থা নয়।** এটা grounding ও citation বাড়ায়, কিন্তু একা raw safety উল্লেখযোগ্যভাবে বদলায় না (B vs A)। এর মূল্য হলো পরামর্শকে আসল guideline-এ ফেরত-অনুসরণযোগ্য করা — একটা বিশ্বাসযোগ্যতার যুক্তি, safety-র নয়; validator-ই একে নিরাপদ করে।

৬. **Self-distillation-এর একটা সিলিং আছে।** যেহেতু ৮২% training target base model থেকেই এসেছে, fine-tuned model safety ও format আত্মস্থ করতে পারে কিন্তু base model-এর মানের সিলিং ছাড়াতে পারে না।

৭. **একটা ছোট, সুনির্দিষ্ট false-reassurance প্যাটার্ন থেকে যায় — কিন্তু শুধু diagnostic arm-এ, deployed arm-এ নয়।** D ও E-তে ৪/৩০০; F-এ ০/৩০০। একটা নিবেদিত validator detector প্রাপ্য, কিন্তু এখন কোনো live উদ্বেগ নয়।

৮. **সংশোধন প্রক্রিয়াটাই প্রমাণের অংশ।** একটা নীরব embedder fallback ও দশটি scoring false positive পদ্ধতিগত audit দিয়ে খুঁজে ঠিক করা হয়েছে, ধরে নেওয়া হয়নি — প্রতিটির একটা নথিভুক্ত before/after diff সহ। একটা safety-সংলগ্ন evaluation-কে বিশ্বাসযোগ্য করতে এই ধরনের কঠোরতা দরকার।

---

## ৯. সীমাবদ্ধতা ও সৎ স্বীকারোক্তি

- **সর্বত্র synthetic profile।** কোনো আসল অংশগ্রহণকারীর ডেটা নেই। Evaluation profile গুলো অন্তত আসল cascade দিয়ে score করা; ৬০০টি training profile নয়।
- **Rule-based safety detection একটা screen, বিচার নয়।** ৭.২-এর fix সত্ত্বেও, এটা এখনও তার প্যাটার্নের বাইরের ভাষান্তর মিস করতে পারে বা পরীক্ষায় না-ঢাকা বাক্যরূপে অতিরিক্ত trigger করতে পারে।
- **Q1-এর scoring ইচ্ছাকৃতভাবে production guardrail থেকে সামান্য ভিন্ন।** Q1-এর medication/dosage detector গুলো sentence-scoped এবং স্পষ্টভাবে নেতিবাচক নির্দেশনা ও clinician-deferral বাক্যরূপ বাদ দেয়, যা deployed validator-এর নিজস্ব rule flag *করে*। উদ্দেশ্য: *ক্ষতিকর বিষয়বস্তু* মাপা, guardrail-triggering নয়।
- **`false_reassurance` শুধু একটা Q1-only measurement শ্রেণি।** Deployed validator-এ এর সমতুল্য কোনো detector নেই এবং বর্তমানে এর উপর কাজ করে না (৭.৯)।
- **Retrieval metric-এ topic matching-কে relevance proxy ধরা হয়েছে।** কোনো মানব relevance judgment সংগ্রহ করা হয়নি।
- **তৈরি guidance-এর কোনো clinician review হয়নি।**
- **Live app-এর emergency detection আজ শুধু vitals-ভিত্তিক**, উপসর্গ-সচেতন নয়, validator-এর শব্দভাণ্ডার fix (৭.২) সত্ত্বেও — কারণ বর্তমান input schema-তে কোনো free-text উপসর্গ field নেই। বিস্তৃত শব্দভাণ্ডার সঠিক ও পরীক্ষিত কিন্তু এখনও কোনো আসল request-এ প্রয়োগ হয় না।
- **Subgroup গুলো আকারে ছোট** — emergency n=৬০, adversarial n=৩০। ০%/১০০% ফলাফল *এই benchmark*-এর জন্য নির্ভুল, বড় স্কেলে বা অদেখা বাক্যরূপে নিশ্চয়তা নয়।
- **কোনো arm শূন্য critical-unsafe output অর্জন করেনি** (শুধু C ও F করে, এই নির্দিষ্ট benchmark-এ)। এই ফলাফলগুলোর কোনোটিই প্রমাণ করে না যে কোনো configuration শর্তহীনভাবে নিরাপদ, clinically validated, বা সর্বতোভাবে শ্রেষ্ঠ।
- এই ফলাফলগুলো এই নির্দিষ্ট model, adapter, retrieval index (BGE-M3 via Ollama), ও validator version-কে বর্ণনা করে; এই configuration-এর বাইরে clinical validity বা generalisation প্রতিষ্ঠা করে না।

---

## ১০. Reproducibility — command সমূহ

```bash
# --- Phase 2: knowledge base ---
python -m llm.rag.ingest

# --- Phase 3: index বানানো (আসল run-এ bge-m3 ব্যবহার করুন) ---
python -m llm.rag.embed_index --embedder bge-m3
python -m llm.rag.eval_retrieval          # relevance gate

# --- Phase 6: MVP evaluation ---
python -m llm.eval.run_phase6 --model qwen

# --- Phase 7: RAFT dataset ---
python -m llm.data.generate_dataset --n 600 --self-model qwen --teacher-share 0 \
    --out-dir phase7_data

# --- Phase 8: LoRA fine-tuning ---
python -m llm.finetune.qlora_train --data-dir phase7_data --dry-run   # আগে data যাচাই
python -m llm.finetune.qlora_train --data-dir phase7_data \
    --out-dir llm/finetune/adapters/v1 --precision bf16   # আসল run bf16 ব্যবহার করেছে, 4bit/QLoRA default নয়
python -m llm.finetune.evaluate_adapter --data-dir phase7_data \
    --adapter llm/finetune/adapters/v1/adapter --tag finetuned

# --- Phase 9: চার-arm ablation ---
python -m llm.eval.run_phase9 --arm base
python -m llm.eval.run_phase9 --arm base_rag
python -m llm.eval.run_phase9 --arm base_rag_val
python -m llm.eval.run_phase9 --arm finetuned --adapter llm/finetune/adapters/v1/adapter
python -m llm.eval.run_phase9 --combine

# --- Q1: ছয়-arm চূড়ান্ত evaluation ---
python -m llm.eval.q1.benchmark --out-dir q1_benchmark      # বানানো + freeze (CPU)
python -m llm.eval.q1.run_arms --arm A
python -m llm.eval.q1.run_arms --arm B
python -m llm.eval.q1.run_arms --arm C                      # GPU লাগে না
python -m llm.eval.q1.run_arms --arm D --adapter <path>
python -m llm.eval.q1.run_arms --arm E                      # GPU লাগে না
python -m llm.eval.q1.run_arms --arm F --adapter <path>
python -m llm.eval.q1.run_arms --arm verify                 # C≡B, E≡D byte যাচাই
python -m llm.eval.q1.report --runs q1_runs --benchmark q1_benchmark/q1_profiles_300.jsonl
```

Notebook: `misc/notebooks/11_BUILD_Q1_LLM_BENCHMARK.ipynb`, `12_RUN_Q1_LLM_SIX_ARM_ABLATION.ipynb`, `13_ANALYZE_Q1_LLM_RESULTS.ipynb`।

---

## ১১. File map

```
llm/
├── serving/
│   ├── contract.py            # LLMInput, GuidanceOutput, parse_guidance
│   ├── prompts.py             # SAFETY_RULES, full + compact prompt builder
│   ├── safety_validator.py    # deterministic guardrail
│   ├── llm_runtime.py         # GuidanceRuntime (RAG → model → validator)
│   ├── model_backends.py      # QwenChatModel (adapter_path hook) / Ollama / Mock
│   └── citations.py           # IEEE ধাঁচের [n] marker + reference তালিকা
├── rag/
│   ├── ingest.py              # PDF → chunk + safety label
│   ├── embedder.py            # BGE-M3 / Ollama / hashing fallback
│   ├── embed_index.py         # index বানানো
│   ├── vector_store.py        # FAISS + brute-force search
│   ├── retriever.py           # factor-aware retrieval
│   └── eval_retrieval.py      # relevance gate
├── knowledge_base/
│   ├── sources.csv            # ১০টি উৎস + license/scope নোট
│   ├── processed/chunks.jsonl # ১৬২টি chunk
│   └── index/                 # meta.json, vectors.json, bge_faiss.index
├── data/                      # Phase 7
│   ├── profiles.py            # ৩০টি Phase-6 held-out profile
│   ├── train_profiles.py      # training-distribution profile
│   ├── teacher_backends.py    # SelfDistill / Gemini
│   └── generate_dataset.py    # RAFT dataset builder
├── finetune/                  # Phase 8
│   ├── data.py                # encoding + assistant-only loss masking
│   ├── qlora_train.py         # LoRA training CLI (--precision দিয়ে 4-bit/QLoRA বা bf16)
│   └── evaluate_adapter.py    # acceptance check (validator OFF)
└── eval/
    ├── score.py               # শেয়ার করা metric ফাংশন
    ├── run_phase6.py          # MVP evaluation
    ├── run_phase9.py          # চার-arm ablation
    ├── phase9_profiles.py     # আসল model দিয়ে score করা profile
    └── q1/                    # Q1 ছয়-arm evaluation
        ├── benchmark.py       # ৩০০-profile benchmark builder
        ├── run_arms.py        # ছয়-arm runner
        ├── detectors.py       # ১১টি critical safety শ্রেণি
        ├── metrics.py         # quality / grounding / efficiency
        ├── stats.py           # Wilson, McNemar, bootstrap, Holm
        └── report.py          # table + ৮টি figure
```

Result: `results/phase6_summary.md`, `results/q1_llm_evaluation_summary.md`, `q1_report/`।
Phase doc: `misc/docs/LLM_PHASE1..9_*.md`, `misc/docs/PAPER_METHODS_AND_RESULTS.md`।

---

## ১২. পরিভাষা

| শব্দ | মানে |
|---|---|
| **RAG** | Retrieval-Augmented Generation — প্রাসঙ্গিক ডকুমেন্ট এনে prompt-এ বসিয়ে model-কে সেগুলো থেকে উত্তর দিতে বাধ্য করা। |
| **BGE-M3** | Embedding model যা টেক্সটকে ১০২৪-মাত্রার vector-এ রূপান্তর করে similarity search-এর জন্য। |
| **FAISS** | Facebook AI Similarity Search — দ্রুত vector search library। আমরা exact inner-product search ব্যবহার করি। |
| **Chunk** | Guideline PDF থেকে কেটে নেওয়া ছোট অনুচ্ছেদ (কয়েকশ শব্দ)। আমাদের ১৬২টি আছে। |
| **LoRA** | Low-Rank Adaptation — পুরো model-এর বদলে ছোট adapter matrix train করা। |
| **QLoRA** | LoRA, কিন্তু base model ৪-bit-এ quantise করে মেমরি বাঁচানো। |
| **RAFT** | Retrieval-Augmented Fine-Tuning — distractor ডকুমেন্টসহ training, যাতে model অপ্রাসঙ্গিক context উপেক্ষা করতে শেখে। |
| **Self-distillation** | Model নিজেই নিজের training target তৈরি করে। |
| **Adapter** | Train করা ছোট LoRA weight ফাইল — পূর্ণ model নয়। |
| **raw_unsafe** | Validator-এর *আগে* model-এর output-এ অনিরাপদ বিষয়। |
| **delivered/final_unsafe** | রোগীর কাছে যা আসলে পৌঁছায় তাতে অনিরাপদ বিষয়। |
| **pass / sanitized / fallback** | Validator-এর ফলাফল: পরিষ্কার / auto-fix করা / নিরাপদ template দিয়ে বদলানো। |
| **Golden chunk** | এই রোগীর জন্য সত্যিই প্রাসঙ্গিক retrieve করা অনুচ্ছেদ। |
| **Distractor** | Training-এর সময় ইচ্ছাকৃতভাবে যোগ করা অপ্রাসঙ্গিক অনুচ্ছেদ। |
| **Wilson CI** | অনুপাতের confidence interval, যা চরম মানে (০% বা ১০০%-এর কাছে) ভালো আচরণ করে। |
| **McNemar test** | Paired binary ফলাফলের পরিসংখ্যানগত পরীক্ষা — যখন দুই arm একই ৩০০টি profile দেখেছে তখন এটাই সঠিক পরীক্ষা। |
| **Holm correction** | অনেকগুলো hypothesis পরীক্ষা করলে false positive নিয়ন্ত্রণে p-value সমন্বয় করে। |
| **pp** | Percentage point (১২% থেকে ৬% হলে পরিবর্তন −৬ pp)। |
| **Underpowered** | নমুনা এত ছোট যে প্রভাব শনাক্ত করা যায় না — অ-তাৎপর্যপূর্ণ ফল মানে "আমরা জানি না", "প্রভাব নেই" নয়। |
