#!/usr/bin/env python
"""
medgemma_all_27b.py

Run MedGemma 27B (text-only) on a CSV of multiple-choice questions,
using multiple prompt variants. Each prompt variant writes its own CSV.

Repository layout (expected):

    .
    ├── data/
    │   └── questions.csv      # input questions
    ├── medgemma_all_27b.py    # this script
    └── ...

Input CSV: data/questions.csv
Required columns:
    - stem
    - option_A
    - option_B
    - option_C
    - option_D
Optional ID columns (first found will be used):
    - question_id, id, qid, QID, QuestionID

Outputs:
    - medgemma_27b_prompt1_jsonstyle.csv
    - medgemma_27b_prompt2_safer.csv
    - medgemma_27b_prompt3_think.csv
    - medgemma_27b_prompt4_doublecheck.csv

Setup:

    pip install -U "transformers>=4.50.0" accelerate torch pandas

You must have gated access to:
    google/medgemma-27b-text-it
on Hugging Face.

NOTE: This is a 27B model. Realistically requires a strong GPU setup
(device_map="auto" will try to shard across available GPUs).
"""

# ------------------------------------------------------
# 0. Imports + OpenMP workaround for macOS
# ------------------------------------------------------
import os

# Fix for: "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import re
import json
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM


# ------------------------------------------------------
# 1. Basic config
# ------------------------------------------------------

# MedGemma 27B text-only, instruction-tuned
MODEL_ID = "google/medgemma-27b-text-it"

# Input CSV with your questions (relative to repo root)
CSV_PATH = Path("data/questions.csv").resolve()

# Each prompt variant will write its own CSV, e.g.:
#   medgemma_27b_prompt1_jsonstyle.csv
OUTPUT_PREFIX = "medgemma_27b"

# Max new tokens for generation
# NOTE: 27B is expensive – adjust based on your hardware.
MAX_NEW_TOKENS = 512

# Small delay between calls (seconds) – adjust or set to 0 for speed
SLEEP_BETWEEN_CALLS = 0.0


# ------------------------------------------------------
# 2. Prompt variants
# ------------------------------------------------------
PROMPTS = {
    "prompt1_jsonstyle": """
You’re a careful clinician. When you answer, you should only give one short, structured reply — not a paragraph, not a sentence, not an explanation. Just a small bundle of information with four parts:

- choice – Pick A, B, C, or D if you’re sure about the answer. If you’re not sure, leave this blank (write "null").
- abstain_code – If you didn’t pick an answer, say why. For example:
  - NO_VALID_OPTION – None of the choices make sense.
  - INSUFFICIENT_INFO – You don’t have enough information.
  - NO_ACTION_NEEDED – Nothing needs to be done.
  Or leave it blank ("null") if you did pick an answer.
- confidence – Say how sure you are, using a number between 0 and 1 (like 0.8 = 80% sure).
- rationale – Give a short, one- to three-sentence reason for your decision.

Important:
- You can’t both pick an answer and say you’re abstaining.
- Either you choose (A–D) or you say why you didn’t.

Respond ONLY in this JSON format and nothing else:

{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}
""",
    "prompt2_safer": """
You’re a careful clinician. When you answer, you should only give one short, structured reply — not a paragraph, not a sentence, not an explanation. Just a small bundle of information with four parts:

- choice – Pick A, B, C, or D if you’re sure about the answer. If you’re not sure, leave this blank (write "null").
- abstain_code – If you didn’t pick an answer, say why. For example:
  - NO_VALID_OPTION – None of the choices make sense.
  - INSUFFICIENT_INFO – You don’t have enough information.
  - NO_ACTION_NEEDED – Nothing needs to be done.
  Or leave it blank ("null") if you did pick an answer.
- confidence – Say how sure you are, using a number between 0 and 1 (like 0.8 = 80% sure).
- rationale – Give a short, one- to three-sentence reason for your decision.

Safety rule:
- If you are not highly confident that one of the options is clearly correct, you MUST abstain.
- In that case, set "choice" = "null" and use an appropriate "abstain_code".
- It is always safer to withhold judgment than to guess.

Important:
- You can’t both pick an answer and say you’re abstaining.
- Either you choose (A–D) or you say why you didn’t.

Respond ONLY in this JSON format and nothing else:

{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}
""",
    "prompt3_think": """
You’re a careful clinician. Before you answer, silently go through three steps:

1. Identify the key clinical findings and what the correct diagnosis or management would be in general.
2. Check whether any of the four options (A–D) truly match that correct diagnosis or management.
3. If none of the options are appropriate, you must abstain instead of guessing.

Do this reasoning in your head and do NOT show these steps. After thinking, give only one short, structured reply with four parts:

- choice – Pick A, B, C, or D if, after checking, you are sure one option is appropriate. If none are appropriate or you are unsure, leave this blank (write "null").
- abstain_code – If you didn’t pick an answer, say why. For example:
  - NO_VALID_OPTION – None of the choices make sense.
  - INSUFFICIENT_INFO – You don’t have enough information.
  - NO_ACTION_NEEDED – Nothing needs to be done.
  Or leave it blank ("null") if you did pick an answer.
- confidence – Say how sure you are, using a number between 0 and 1 (like 0.8 = 80% sure).
- rationale – Give a short, one- to three-sentence reason for your decision.

Important:
- You can’t both pick an answer and say you’re abstaining.
- Either you choose (A–D) or you say why you didn’t.

Respond ONLY in this JSON format and nothing else:

{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}
""",
    "prompt4_doublecheck": """
You’re a careful clinician. Before you answer, silently go through two steps:

1. Tentatively decide which option (A–D), if any, seems best based on the clinical vignette.
2. Double-check that decision by asking yourself:
   - “Is it possible that none of these options are actually appropriate for this patient?”
   - If it is possible that none are appropriate, you must abstain instead of guessing.

Do this reasoning in your head and do NOT show these steps. After thinking, give only one short, structured reply with four parts:

- choice – Pick A, B, C, or D only if, after double-checking, you are confident that at least one option is truly appropriate. If you judge that none are appropriate or you are unsure, leave this blank (write "null").
- abstain_code – If you didn’t pick an answer, say why. For example:
  - NO_VALID_OPTION – None of the choices make sense.
  - INSUFFICIENT_INFO – You don’t have enough information.
  - NO_ACTION_NEEDED – Nothing needs to be done.
  Or leave it blank ("null") if you did pick an answer.
- confidence – Say how sure you are, using a number between 0 and 1 (like 0.8 = 80% sure).
- rationale – Give a short, one- to three-sentence reason for your decision.

Important:
- You can’t both pick an answer and say you’re abstaining.
- Either you choose (A–D) or you say why you didn’t.

Respond ONLY in this JSON format and nothing else:

{
  "choice": "A/B/C/D or null",
  "abstain_code": "NO_VALID_OPTION/INSUFFICIENT_INFO/NO_ACTION_NEEDED or null",
  "confidence": 0.0,
  "rationale": "one to three sentences"
}
"""
}


# ------------------------------------------------------
# 3. Robust JSON-ish parser
# ------------------------------------------------------
def safe_parse_json(text: str) -> dict:
    """
    Try very hard to recover a structured answer from model output.

    Returns a dict with keys:
      - choice
      - abstain_code
      - confidence
      - rationale
    """
    default = {
        "choice": None,
        "abstain_code": "INSUFFICIENT_INFO",
        "confidence": 0.0,
        "rationale": (
            "Model did not respond in the requested JSON-like format; "
            "treating as abstention."
        ),
    }

    if not isinstance(text, str) or not text.strip():
        return default

    # Remove <think> blocks (if any)
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # Remove markdown code fences
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    cleaned = cleaned.strip()

    # ---------- 1) Try full JSON ----------
    obj = None
    try:
        obj = json.loads(cleaned)
    except Exception:
        obj = None

    # ---------- 2) Try substring JSON ----------
    if obj is None:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = cleaned[start : end + 1]
            try:
                obj = json.loads(candidate)
            except Exception:
                obj = None

    # ---------- 3) If JSON parsing succeeded ----------
    if isinstance(obj, dict):
        choice = obj.get("choice")
        abstain = obj.get("abstain_code")
        conf = obj.get("confidence")
        rationale = obj.get("rationale")

        # Normalize "null"/"" -> None
        if isinstance(choice, str) and choice.strip().lower() in {"null", ""}:
            choice = None
        if isinstance(abstain, str) and abstain.strip().lower() in {"null", ""}:
            abstain = None

        # Normalize confidence
        try:
            conf = float(conf) if conf is not None else 0.0
        except Exception:
            conf = 0.0

        return {
            "choice": choice,
            "abstain_code": abstain,
            "confidence": conf,
            "rationale": rationale,
        }

    # ---------- 4) Regex-based extraction fallback ----------
    flat = " ".join(cleaned.split())

    # choice
    choice = None
    m_choice = re.search(
        r'["\']?\s*choice\s*["\']?\s*:\s*["\']?([ABCD]|null)["\']?',
        flat,
        re.IGNORECASE,
    )
    if m_choice:
        raw_choice = m_choice.group(1)
        choice = None if raw_choice.lower() == "null" else raw_choice.upper()

    # abstain_code
    abstain = None
    m_abstain = re.search(
        r'["\']?\s*abstain_code\s*["\']?\s*:\s*["\']?([A-Z_]+|null)["\']?',
        flat,
        re.IGNORECASE,
    )
    if m_abstain:
        raw_abstain = m_abstain.group(1)
        abstain = None if raw_abstain.lower() == "null" else raw_abstain.upper()

    # confidence
    conf = 0.0
    m_conf = re.search(
        r'["\']?\s*confidence\s*["\']?\s*:\s*([0-9]*\.?[0-9]+)',
        flat,
        re.IGNORECASE,
    )
    if m_conf:
        try:
            conf = float(m_conf.group(1))
        except Exception:
            conf = 0.0

    # rationale
    rationale = None
    m_rat = re.search(
        r'["\']?\s*rationale\s*["\']?\s*:\s*["\'](.*?)["\']',
        cleaned,
        re.DOTALL | re.IGNORECASE,
    )
    if m_rat:
        rationale = m_rat.group(1).strip()

    if choice is not None or abstain is not None or rationale is not None or conf != 0.0:
        return {
            "choice": choice,
            "abstain_code": abstain,
            "confidence": conf,
            "rationale": rationale
            or "Recovered fields from malformed JSON-like output.",
        }

    return default


# ------------------------------------------------------
# 4. Helper: leave CSV cells empty instead of "null"
# ------------------------------------------------------
def to_csv_null(v):
    """
    For CSV export:
      - None or "null" (case-insensitive) → empty cell
      - else: return value as-is
    """
    if v is None:
        return ""
    if isinstance(v, str) and v.strip().lower() == "null":
        return ""
    return v


# ------------------------------------------------------
# 5. Build question text
# ------------------------------------------------------
def build_question_block(row: pd.Series) -> str:
    stem = row["stem"]
    opts = (
        f"A. {row['option_A']}\n"
        f"B. {row['option_B']}\n"
        f"C. {row['option_C']}\n"
        f"D. {row['option_D']}"
    )
    return f"{stem}\n\nOptions:\n{opts}"


# ------------------------------------------------------
# 6. Load questions
# ------------------------------------------------------
def load_questions(csv_path: Path) -> Tuple[pd.DataFrame, Optional[str]]:
    print(f"Using questions CSV at: {csv_path}")
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found at {csv_path}")

    df = pd.read_csv(csv_path)

    required_cols = ["stem", "option_A", "option_B", "option_C", "option_D"]
    for col in required_cols:
        if col not in df.columns:
            raise SystemExit(f"Missing required column: {col}")

    # Prefer 'question_id' if exists
    id_col = None
    for candidate in ["question_id", "id", "qid", "QID", "QuestionID"]:
        if candidate in df.columns:
            id_col = candidate
            break

    print("Detected ID column:", id_col if id_col else "(none, will use row index)")
    print(f"Total questions: {len(df)}\n")
    return df, id_col


# ------------------------------------------------------
# 7. Load MedGemma 27B model & tokenizer
# ------------------------------------------------------
def load_model_and_tokenizer():
    print("Loading MedGemma 27B (this may take a while)...")

    # IMPORTANT: This expects a GPU setup. On CPU or small GPUs this will likely OOM.
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",  # let HF / accelerate shard across available GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    print("Model and tokenizer loaded.")
    return model, tokenizer


# ------------------------------------------------------
# 8. Call MedGemma once
# ------------------------------------------------------
def call_medgemma(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_instructions: str,
    user_prompt: str,
) -> str:
    """
    Build chat-style messages, apply chat template, run generate(), and decode.
    Uses the recommended pattern from the MedGemma model card.
    """
    messages = [
        {"role": "system", "content": system_instructions},
        {"role": "user", "content": user_prompt},
    ]

    try:
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        # Fallback: plain concatenation if chat template fails
        merged = f"{system_instructions.strip()}\n\n{user_prompt.strip()}"
        inputs = tokenizer(
            merged,
            return_tensors="pt",
        ).to(model.device)

    # If we used apply_chat_template with return_dict, inputs is a dict
    if isinstance(inputs, dict):
        input_len = inputs["input_ids"].shape[-1]
    else:
        input_len = inputs.shape[-1]

    try:
        with torch.inference_mode():
            if isinstance(inputs, dict):
                generation = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )
            else:
                generation = model.generate(
                    inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

        # Keep only the newly generated tokens (completion after the prompt)
        generated_ids = generation[0][input_len:]
        decoded = tokenizer.decode(generated_ids, skip_special_tokens=True)
        return decoded.strip()

    except Exception as e:
        print(f"\nMedGemma generate failed: {e}")
        return f"ERROR: generate failed: {e}"


# ------------------------------------------------------
# 9. Main loop: run all prompts and write CSVs
# ------------------------------------------------------
def run_all_prompts():
    df, id_col = load_questions(CSV_PATH)
    total = len(df)

    model, tokenizer = load_model_and_tokenizer()

    for prompt_key, instructions in PROMPTS.items():
        output_path = f"{OUTPUT_PREFIX}_{prompt_key}.csv"
        print("=" * 60)
        print(f"Running prompt variant: {prompt_key}")
        print(f"Output file will be: {output_path}\n")

        rows_out = []

        for idx, row in df.iterrows():
            qid = row[id_col] if id_col else idx + 1
            user_prompt = build_question_block(row)

            print(f"  → Question {idx + 1}/{total} (ID: {qid}) ...", end="", flush=True)

            med_raw = call_medgemma(
                model=model,
                tokenizer=tokenizer,
                system_instructions=instructions.strip(),
                user_prompt=user_prompt,
            )

            if isinstance(med_raw, str) and med_raw.startswith("ERROR:"):
                med_parsed = {
                    "choice": None,
                    "abstain_code": "API_ERROR",
                    "confidence": 0.0,
                    "rationale": med_raw,
                }
                print(" ERROR")
            else:
                med_parsed = safe_parse_json(med_raw)
                print(" done")

            out_row = {
                "row_index": idx + 1,
                "question_id": qid,
                "stem": row["stem"],
                "option_A": row["option_A"],
                "option_B": row["option_B"],
                "option_C": row["option_C"],
                "option_D": row["option_D"],
                "medgemma_raw": med_raw,
                "medgemma_choice": to_csv_null(med_parsed["choice"]),
                "medgemma_abstain_code": to_csv_null(med_parsed["abstain_code"]),
                "medgemma_confidence": (
                    med_parsed["confidence"]
                    if med_parsed["confidence"] is not None
                    else 0.0
                ),
                "medgemma_rationale": to_csv_null(med_parsed["rationale"]),
            }

            rows_out.append(out_row)
            if SLEEP_BETWEEN_CALLS > 0:
                time.sleep(SLEEP_BETWEEN_CALLS)

        pd.DataFrame(rows_out).to_csv(output_path, index=False)
        print(f"\nFinished {prompt_key}. Saved {len(rows_out)} rows to {output_path}\n")

    print("All prompts completed.")


# ------------------------------------------------------
# 10. Entry point
# ------------------------------------------------------
if __name__ == "__main__":
    run_all_prompts()
