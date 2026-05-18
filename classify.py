import os
import re
import json
import time
import ollama
import pandas as pd

df = pd.read_csv('iclr2026_papers.csv')

CSV_PATH = 'safety_results.csv'
COLUMNS = ["id", "title", "is_safety", "subdomain", "confidence", "evidence"]

if os.path.exists(CSV_PATH):
    with open(CSV_PATH, "r", encoding="utf-8", errors="ignore") as _f:
        already_done = max(0, sum(1 for _ in _f) - 1)
    write_header = False
else:
    already_done = 0
    write_header = True

print(f"Resuming from paper {already_done + 1}")


SYSTEM_PROMPT = """
You are an expert AI safety researcher classifying academic papers.

A paper is AI safety ONLY if making AI systems safer, more aligned, more honest, or less harmful is a PRIMARY contribution. A passing mention or a generic "robustness" / "alignment" / "agent" claim does NOT count.

SUBDOMAINS (use these EXACT strings):
- Alignment — RLHF, DPO, GRPO, preference learning, reward modeling, constitutional AI, value alignment of LLMs
- Interpretability — mechanistic interpretability, circuit analysis, feature attribution AIMED at understanding model internals for safety/transparency
- Robustness & Adversarial — adversarial attacks/defenses on AI systems, certified robustness, jailbreak-style perturbations
- Hallucination & Factuality — reducing hallucination, factuality, calibration, faithfulness of LLM outputs
- Agent Safety — risks of autonomous LLM agents, prompt injection, tool-use safety
- Evaluation & Benchmarking — benchmarks WHOSE PURPOSE is measuring safety, harm, bias, fairness, toxicity
- Deception & Scheming — jailbreaking, red teaming, backdoor attacks, deceptive alignment
- Privacy & Security — differential privacy, membership inference, data poisoning, model extraction
- Societal & Governance — AI governance, regulation, societal impact, responsible AI
- Not AI Safety — anything else

STRICT EXCLUSIONS — these are NOT AI safety:
✗ "Alignment" of image/text/modality/features/representations/cross-lingual — this is representation learning, NOT safety
✗ General model robustness or OOD generalization for accuracy gains
✗ RL agents, multi-agent RL, game-playing agents, agent-based simulation
✗ Generic "evaluation" or "benchmark" papers that don't target safety/harm/bias
✗ Calibration for prediction accuracy (not for LLM faithfulness)
✗ Standard regularization or noise robustness
✗ Papers that merely mention safety in motivation but don't contribute to it

DECISION RULE: When in doubt, output "Not AI Safety". False positives are worse than false negatives.

OUTPUT FORMAT — JSON array only, no prose. Each object MUST have:
- "paper_id": int
- "is_safety": bool
- "subdomain": one of the exact strings above
- "confidence": "high" | "medium" | "low"
- "evidence": a SHORT VERBATIM quote (≤15 words) from the abstract that proves safety is a primary contribution. Empty string "" if not safety.

EXAMPLES:

Title: Direct Preference Optimization with Length Normalization
Abstract: We propose a length-normalized DPO objective to reduce verbosity bias in RLHF-tuned models...
→ {"paper_id": 1, "is_safety": true, "subdomain": "Alignment", "confidence": "high", "evidence": "length-normalized DPO objective to reduce verbosity bias in RLHF"}

Title: Cross-Modal Alignment for Vision-Language Pretraining
Abstract: We align image and text features in a shared embedding space to improve retrieval...
→ {"paper_id": 2, "is_safety": false, "subdomain": "Not AI Safety", "confidence": "high", "evidence": ""}

Title: Multi-Agent Reinforcement Learning for Traffic Control
Abstract: We train cooperative agents to optimize traffic signal timing...
→ {"paper_id": 3, "is_safety": false, "subdomain": "Not AI Safety", "confidence": "high", "evidence": ""}

Title: Prompt Injection Attacks on Tool-Using LLM Agents
Abstract: We show that adversarial tool descriptions can hijack agent behavior, and propose a defense...
→ {"paper_id": 4, "is_safety": true, "subdomain": "Agent Safety", "confidence": "high", "evidence": "adversarial tool descriptions can hijack agent behavior, and propose a defense"}

Title: Robust Training under Label Noise
Abstract: We propose a loss that improves accuracy when training labels are corrupted...
→ {"paper_id": 5, "is_safety": false, "subdomain": "Not AI Safety", "confidence": "high", "evidence": ""}

Title: A Benchmark for Measuring Toxicity in Multilingual LLMs
Abstract: We release TOXBENCH, covering 30 languages, to evaluate toxic generation in LLMs...
→ {"paper_id": 6, "is_safety": true, "subdomain": "Evaluation & Benchmarking", "confidence": "high", "evidence": "evaluate toxic generation in LLMs"}

Respond ONLY as a JSON array. No explanation. No extra text. Just JSON.
"""

VALID_SUBDOMAINS = [
    "Alignment", "Interpretability", "Robustness & Adversarial",
    "Hallucination & Factuality", "Agent Safety", "Evaluation & Benchmarking",
    "Deception & Scheming", "Privacy & Security", "Societal & Governance",
    "Not AI Safety",
]
def normalize_subdomain(s):
    if s is None:
        return "Not AI Safety"
    s = str(s).strip()
    low = s.lower()
    for v in VALID_SUBDOMAINS:
        if low == v.lower():
            return v
    return "Not AI Safety"


def normalize_confidence(c):
    if c is None:
        return "low"
    c = str(c).lower().strip()
    if "high" in c:
        return "high"
    if "med" in c:
        return "medium"
    if "low" in c:
        return "low"
    return "low"


def normalize_is_safety(v, subdomain, confidence, evidence, abstract):
    if subdomain == "Not AI Safety":
        return False
    if isinstance(v, bool):
        flag = v
    elif isinstance(v, (int, float)):
        flag = bool(v)
    elif isinstance(v, str):
        flag = v.strip().lower() in ("true", "1", "yes", "y", "t")
    else:
        flag = True
    if not flag:
        return False
    if confidence != "high":
        return False
    ev = (evidence or "").strip().strip('"\'')
    if len(ev) < 5:
        return False
    abs_low = (abstract or "").lower()
    ev_low = ev.lower()
    if ev_low in abs_low:
        return True
    ev_words = [w for w in re.findall(r"\w+", ev_low) if len(w) > 3]
    if not ev_words:
        return False
    hits = sum(1 for w in ev_words if w in abs_low)
    return hits / len(ev_words) >= 0.6


def extract_json_array(text):
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```\s*$", "", text)

    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("results", "papers", "classifications", "data", "output"):
                if key in data and isinstance(data[key], list):
                    return data[key]
            if all(k in data for k in ("paper_id",)) or "subdomain" in data:
                return [data]
    except Exception:
        pass

    m = re.search(r"\[[\s\S]*\]", text)
    if m:
        snippet = m.group(0)
        try:
            data = json.loads(snippet)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        try:
            cleaned = re.sub(r",\s*([\]}])", r"\1", snippet)
            data = json.loads(cleaned)
            if isinstance(data, list):
                return data
        except Exception:
            pass

    objs = []
    depth = 0
    start = None
    for i, ch in enumerate(text):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = text[start:i + 1]
                try:
                    objs.append(json.loads(chunk))
                except Exception:
                    try:
                        objs.append(json.loads(re.sub(r",\s*([\]}])", r"\1", chunk)))
                    except Exception:
                        pass
                start = None
    if objs:
        return objs
    return None


def build_user_prompt(papers):
    prompt = ""
    for idx, row in enumerate(papers):
        title = row.get("title", "") if isinstance(row, dict) else row["title"]
        abstract = row.get("abstract", "") if isinstance(row, dict) else row["abstract"]
        if pd.isna(title):
            title = ""
        if pd.isna(abstract):
            abstract = ""
        title = str(title)[:500]
        abstract = str(abstract)[:1000]
        prompt += f"\nPaper {idx + 1}:\nTitle: {title}\nAbstract: {abstract}\n---\n"
    prompt += f"""
Respond with a JSON array of exactly {len(papers)} objects, one per paper, in order.
"subdomain" MUST be EXACTLY one of these strings:
"Alignment" | "Interpretability" | "Robustness & Adversarial" | "Hallucination & Factuality" | "Agent Safety" | "Evaluation & Benchmarking" | "Deception & Scheming" | "Privacy & Security" | "Societal & Governance" | "Not AI Safety"

"evidence" MUST be a verbatim quote (≤15 words) from the abstract proving safety is a primary contribution, or "" if not safety.
Be STRICT. When in doubt, output "Not AI Safety" with empty evidence.

[
  {{"paper_id": 1, "is_safety": true, "subdomain": "Alignment", "confidence": "high", "evidence": "we propose an RLHF method to reduce harmful outputs"}},
  {{"paper_id": 2, "is_safety": false, "subdomain": "Not AI Safety", "confidence": "high", "evidence": ""}}
]
"""
    return prompt


def call_model(papers, max_attempts=8):
    user_prompt = build_user_prompt(papers)
    last_err = None
    for attempt in range(max_attempts):
        try:
            response = ollama.chat(
                model="hermes3:8b",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                format="json",
                options={"temperature": 0.0 if attempt == 0 else 0.2 + 0.1 * (attempt % 4)},
            )
            content = response.get("message", {}).get("content", "")
            data = extract_json_array(content)
            if data is None:
                last_err = "json parse failed"
            elif len(data) == 0:
                last_err = "empty array"
            else:
                return data
        except Exception as e:
            last_err = str(e)
        print(f"    attempt {attempt + 1}/{max_attempts} failed ({last_err}), retrying...")
        time.sleep(min(2 ** attempt, 15))
    return None


JUDGE_SYSTEM = (
    "You are a strict AI safety classifier. "
    "Reply with 'yes' or 'no' on the first line, then one short sentence."
)


def judge_safety(title, abstract, subdomain, evidence, max_attempts=3):
    if pd.isna(title):
        title = ""
    if pd.isna(abstract):
        abstract = ""
    title = str(title)[:300]
    abstract = str(abstract)[:1000]
    prompt = f"""A paper was classified as AI safety related under subdomain "{subdomain}".

Title: {title}
Abstract: {abstract}
Evidence given: {evidence}

Is AI safety truly the PRIMARY contribution of this paper, or is it just a passing mention or unrelated use of similar terminology (e.g., "alignment" of features, RL "agents", general "robustness")?

Answer with ONLY: yes or no
Then one sentence explaining why."""
    last_err = None
    for attempt in range(max_attempts):
        try:
            response = ollama.chat(
                model="hermes3:8b",
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                options={"temperature": 0.0, "num_predict": 40},
            )
            answer = response.get("message", {}).get("content", "").strip().lower()
            answer = answer.lstrip("`*\"' \n\t")
            if answer.startswith("yes"):
                return True
            if answer.startswith("no"):
                return False
            return True
        except Exception as e:
            last_err = str(e)
            time.sleep(min(2 ** attempt, 10))
    print(f"  judge failed ({last_err}); keeping classification")
    return True


def classify_batch(rows):
    papers = [r for _, r in rows]
    n = len(papers)

    data = call_model(papers, max_attempts=6)
    if data is not None and len(data) >= n:
        return data[:n]
    if data is not None and len(data) > 0:
        print(f"  partial response ({len(data)}/{n}), filling rest individually...")
        out = list(data[:n])
    else:
        out = []

    start = len(out)
    for j in range(start, n):
        print(f"  individual classification for paper {j + 1}/{n}...")
        single = call_model([papers[j]], max_attempts=5)
        if single and len(single) >= 1:
            entry = single[0]
            entry["paper_id"] = j + 1
            out.append(entry)
        else:
            print(f"  paper {j + 1} unrecoverable, defaulting to Not AI Safety/low")
            out.append({
                "paper_id": j + 1,
                "is_safety": False,
                "subdomain": "Not AI Safety",
                "confidence": "low",
            })
    return out


BATCH_SIZE = 5
total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

for i in range(already_done, len(df), BATCH_SIZE):
    batch_df = df.iloc[i:i + BATCH_SIZE]
    batch_rows = list(batch_df.iterrows())
    batch_num = i // BATCH_SIZE + 1

    try:
        batch_results = classify_batch(batch_rows)
    except Exception as e:
        print(f"Unexpected error in batch {batch_num}: {e}; defaulting entire batch")
        batch_results = [
            {"paper_id": j + 1, "is_safety": False, "subdomain": "Not AI Safety", "confidence": "low"}
            for j in range(len(batch_rows))
        ]

    batch_out = []
    judge_flipped = 0
    for j, (_, row) in enumerate(batch_rows):
        entry = batch_results[j] if j < len(batch_results) else {}
        subdomain = normalize_subdomain(entry.get("subdomain"))
        confidence = normalize_confidence(entry.get("confidence"))
        evidence = str(entry.get("evidence") or "").strip()
        abstract = "" if pd.isna(row.get("abstract")) else str(row.get("abstract", ""))
        is_safety = normalize_is_safety(
            entry.get("is_safety"), subdomain, confidence, evidence, abstract,
        )
        if is_safety:
            if not judge_safety(row["title"], abstract, subdomain, evidence):
                is_safety = False
                judge_flipped += 1
        if not is_safety:
            subdomain = "Not AI Safety"
            evidence = ""
        batch_out.append({
            "id": row["id"],
            "title": row["title"],
            "is_safety": is_safety,
            "subdomain": subdomain,
            "confidence": confidence,
            "evidence": evidence,
        })

    new_df = pd.DataFrame(batch_out, columns=COLUMNS)
    new_df.to_csv(CSV_PATH, mode="a", header=write_header, index=False)
    write_header = False
    batch_kept = sum(1 for r in batch_out if r["is_safety"])
    batch_out.clear()
    del new_df

    print(
        f"Processed batch {batch_num}/{total_batches} "
        f"(rows {i + 1}-{i + len(batch_rows)}, "
        f"safety kept={batch_kept}, judge flipped={judge_flipped})"
    )

import csv
safety_total = 0
with open(CSV_PATH, "r", encoding="utf-8", errors="ignore", newline="") as _f:
    reader = csv.DictReader(_f)
    for r in reader:
        if str(r.get("is_safety", "")).strip().lower() == "true":
            safety_total += 1
print(f"Done! Total AI safety papers: {safety_total}")
