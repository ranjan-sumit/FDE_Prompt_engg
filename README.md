# FDE Academy — Prompt Engineering Live Demo

## Setup (2 minutes)

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

## First time

1. Paste your NVIDIA API key in the **sidebar** (left panel)
2. Select your **Primary Model** (used for Demos 1–3)
3. Pick a tab and hit Run

## 4 Demos at a glance

| Tab | What it shows | Key moment |
|-----|--------------|------------|
| 🥊 Prompt Battle Arena | Zero-Shot vs Few-Shot vs CoT, same ticket | CoT gets it right, Zero-Shot confidently gets it wrong |
| 🔬 Prompt Autopsy | Paste any prompt → 10-element scorecard | Student's "good" prompt scores 4/20 |
| 🛡️ Harness vs Naked | Same model, with/without system prompt | Jailbreak breaks Naked, Harness holds |
| 🤖 Multi-Model Showdown | Same prompt → all 4 models | Pick who wins, discuss WHY |

## Models available (NVIDIA API)

- `openai/gpt-oss-120b` — flagship, non-streaming
- `openai/gpt-oss-20b` — fast, streaming
- `google/gemma-4-31b-it` — Google's Gemma
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` — shows chain-of-thought reasoning

## Teacher tips

**Demo 1 — Prompt Battle**
- Use the default ticket first (double charge)
- Change the ticket text mid-session for a second run
- Ask class: "Which would you trust in production?"

**Demo 2 — Autopsy**
- Start with the pre-loaded bad prompt (scores ~3/20)
- Then paste the "good" prompt example
- The improved version at the bottom is the real teaching moment

**Demo 3 — Harness vs Naked**
- Click the 🔴 JAILBREAK button for the most dramatic contrast
- Then click ✅ Normal to show the harness still works for regular queries
- Point out: same model, same API call — only the system prompt changes

**Demo 4 — Showdown**
- Use the default CTO prompt (it has specific word count + format constraints)
- Models that ignore "max 80 words" or skip bullet formatting lose the vote
- Good discussion starter: why do models behave differently on the same prompt?
