from datasets import load_dataset
import pandas as pd
import random

# Load the MMLU dataset from Hugging Face
# Each record has: question, answer, subject, choices (list of 4)
ds = load_dataset("cais/mmlu", "all")["test"]

ds

# Pick a few subjects (balanced mix of STEM, humanities, applied)
subjects = [
    "computer_security",
    "logical_fallacies",
    "conceptual_physics",
    "abstract_algebra",
    "human_aging"
]

# Filter dataset
subset = [r for r in ds if r["subject"] in subjects]

# Deterministic shuffle
random.seed(42)
random.shuffle(subset)

# Choose how many per subject (e.g. 20 each)
N_PER_SUBJECT = 5
rows = []
seen = {s: 0 for s in subjects}

for r in subset:
    s = r["subject"]
    if s in subjects and seen[s] < N_PER_SUBJECT:
        q = r["question"].strip().replace("\n", " ")
        choices = r["choices"]
        rows.append({
            "subject": s,
            "question": q,
            "A": choices[0],
            "B": choices[1],
            "C": choices[2],
            "D": choices[3],
            "answer": r["answer"],  # e.g., "A" / "B" / "C" / "D"
        })
        seen[s] += 1
    if all(seen[s] >= N_PER_SUBJECT for s in subjects):
        break

df = pd.DataFrame(rows)
df.to_csv("prompts_mmlu_subset.csv", index=False)
print("Saved prompts_mmlu_subset.csv with", len(df), "questions")
print("Subjects:", df['subject'].unique())
