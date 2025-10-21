import argparse, json, subprocess, time
from datetime import datetime
import pandas as pd

# Map your factor values -> actual Ollama tags on the SUT
MODEL_TAGS = {
    "Llama-3 8B Instruct": {
        "fp16": "llama3:8b",
        "q8_0": "llama3:8b-q8_0",   # <- adjust to your local tag
        "q4_0": "llama3:8b-q4_0",   # <- adjust to your local tag
    },
    "Mistral-7B Instruct": {
        "fp16": "mistral:7b",
        "q8_0": "mistral:7b-q8_0",
        "q4_0": "mistral:7b-q4_0",
    },
    "GLM-4 9B Chat": {
        "fp16": "glm4:9b",
        "q8_0": "glm4:9b-q8_0",
        "q4_0": "glm4:9b-q4_0",
    },
}

USER_TEMPLATE = (
    "Question: {question}\n"
    "Options:\nA. {A}\nB. {B}\nC. {C}\nD. {D}\n\n"
    "Respond with only one letter (A, B, C, or D)."
)

def run_ollama(tag, prompt, temperature=0.0):
    cmd = ["ollama", "run", tag, "--json"]
    payload = json.dumps({"prompt": prompt, "options": {"temperature": temperature}})
    start = time.time()
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        p.stdin.write(payload + "\n"); p.stdin.flush()
    except Exception: pass
    out, tok, first = "", 0, None
    for line in p.stdout:
        line = line.strip()
        if not line: continue
        try: obj = json.loads(line)
        except: continue
        if "response" in obj:
            if first is None: first = time.time()
            out += obj["response"]; tok += 1
        if obj.get("done"): break
    p.wait()
    end = time.time()
    t = end - start
    ttfb = (first - start) if first else t
    tps = tok / t if t > 0 else 0.0
    return out.strip(), t, ttfb, tps

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts_csv", required=True)
    ap.add_argument("--model", required=True)         # e.g., "Llama-3 8B Instruct"
    ap.add_argument("--quant", required=True)         # e.g., "fp16" | "q8_0" | "q4_0"
    ap.add_argument("--questions", type=int, default=20)
    ap.add_argument("--out_csv", default=None)
    args = ap.parse_args()

    tag = MODEL_TAGS[args.model][args.quant]
    df  = pd.read_csv(args.prompts_csv)
    # FIXED SET: take the first N (or pre-filter by subject if you want).
    batch = df.head(args.questions).reset_index(drop=True)

    # Tiny warm-up to stabilize clocks
    _ = run_ollama(tag, "Say: READY", temperature=0.0)

    # TODO: start EnergiBridge here on SUT if available (CLI/API). For now, placeholders:
    trial_t0 = time.time()

    rows = []
    for _, row in batch.iterrows():
        prompt = USER_TEMPLATE.format(
            question=row["question"], A=row["A"], B=row["B"], C=row["C"], D=row["D"]
        )
        pred, latency, ttfb, tps = run_ollama(tag, prompt, temperature=0.0)
        letter = (pred[:1].upper() if pred else "")
        correct = int(letter == str(row["answer"]).strip().upper())
        rows.append({
            "timestamp": datetime.utcnow().isoformat(),
            "model": args.model, "quant": args.quant,
            "subject": row.get("subject", ""),
            "gold": row["answer"], "pred": letter, "correct": correct,
            "latency_s": latency, "ttfb_s": ttfb, "tokens_per_s": tps,
        })

    trial_dur = time.time() - trial_t0
    # TODO: stop EnergiBridge; compute trial energy (CPU/GPU). For now, zeros:
    energy_cpu_j, energy_gpu_j = 0.0, 0.0
    per_prompt_cpu = energy_cpu_j / max(1, len(rows))
    per_prompt_gpu = energy_gpu_j / max(1, len(rows))
    for r in rows:
        r["energy_cpu_j"] = per_prompt_cpu
        r["energy_gpu_j"] = per_prompt_gpu
        r["trial_duration_s"] = trial_dur

    if args.out_csv:
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)

    print(json.dumps({"status": "ok", "rows": rows}))

if __name__ == "__main__":
    main()
