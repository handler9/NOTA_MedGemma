# NOTA MedGemma Evaluation Pipeline

This repository contains a single Python script for evaluating a MedGemma model on multiple-choice clinical reasoning questions. The script runs the same dataset through **four different prompts** and writes **four separate CSV files**, one per prompt.

## Repository Structure

- `data/questions.csv` – input dataset of questions  
- `medgemma_4prompts.py` – main script (runs all 4 prompts)  
- `requirements.txt` – Python dependencies  
- `README.md` – project documentation  

## What the Script Does

When you run `medgemma_4prompts.py`, it:

1. Loads all questions from `data/questions.csv`.
2. Sends each question to the MedGemma 27B model using **four different prompts**:

   - **Prompt 1 – Baseline JSON**  
     Returns a JSON object: choice, abstain code, confidence, rationale.

   - **Prompt 2 – Safer JSON**  
     Same JSON structure, but instructs the model to abstain if uncertain.

   - **Prompt 3 – Think → Decide**  
     The model “thinks” silently, then outputs only a clean JSON answer (no chain-of-thought).

   - **Prompt 4 – Double-Check**  
     The model picks a tentative answer, double-checks safety, and abstains if none of the options are safe.

3. Saves the outputs into **four CSV files**, one for each prompt:

   - `medgemma_prompt1_jsonstyle.csv`  
   - `medgemma_prompt2_safer.csv`  
   - `medgemma_prompt3_think.csv`  
   - `medgemma_prompt4_doublecheck.csv`  

Each CSV includes:

- Original question text & answer options  
- Raw model output  
- Parsed fields: `choice`, `abstain_code`, `confidence`, `rationale`  

## Input Format (`data/questions.csv`)

Required columns:

- `stem`  
- `option_A`  
- `option_B`  
- `option_C`  
- `option_D`  
- `question_id` *(optional — auto-generated if missing)*

Example:

```text
question_id,stem,option_A,option_B,option_C,option_D
1,"A 55-year-old man presents with chest pain...",Aspirin,Beta-blocker,Statin,"CT angiography"
