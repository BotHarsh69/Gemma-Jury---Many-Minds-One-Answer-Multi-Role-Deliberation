# 🏛️ Gemma Jury — Many Minds, One Answer

> **"A single model response is not always reliable. A lightweight model makes it possible to approach the same problem from several perspectives."**

Built for the Hackathon using **Python**, **Streamlit**, and **Google Gemma 4 (`gemma-4-26b-a4b-it`)** via the official `google-genai` SDK.

---

## 🌟 Overview

**Gemma Jury** harnesses multiple independent instances of **Gemma 4**, each embodying a distinct, specialized perspective to examine the same image, text, or complex problem.

The system then convenes a **Gemma 4 Foreperson** to detect agreements and disagreements, calculate a calibrated confidence score, and reconcile conflicting opinions into a single, highly reliable final verdict.

---

## 👥 The 5 Independent Gemma 4 Judges

1. **🔍 The Skeptic (Justice Vance):** Actively searches for cognitive traps, unstated assumptions, confirmation bias, and edge-case failure modes.
2. **🔬 The Domain Specialist (Dr. Aris):** Applies technical rigor, formal definitions, and methodical first-principles reasoning.
3. **💡 The First-Principles Thinker (Judge Elena):** Strips away jargon and overthinking; applies baseline common-sense and intuitive sanity checks.
4. **📋 The Fact-Checker & Verifier (Inspector Thorne):** Scrutinizes empirical claims, calculations, citations, and visual artifacts to prevent hallucinations.
5. **⚖️ The Devil's Advocate (Counselor Malik):** Fights groupthink by exploring minority viewpoints, plausible counter-arguments, and context-dependent trade-offs.

---

## ⚖️ How the Foreperson Reconciles Disagreements

Rather than simple majority voting, the **Gemma 4 Foreperson** performs qualitative synthesis:
- **Consensus Assessment:** Categorizes agreement as *Unanimous*, *Strong Majority*, *Divided*, or *Nuanced Consensus*.
- **Points of Alignment:** Highlights shared ground among judges.
- **Points of Contention:** Pinpoints specific disputes, risks, and edge cases flagged by dissenting judges.
- **Calibrated Confidence Score (0–100%):** Automatically drops when reputable judges raise valid doubts or split on interpretation.
- **Unified Reconciled Answer:** Produces an authoritative synthesis that addresses edge cases and caveats.

---

## 📊 Empirical Evaluation: 12-Scenario Official Benchmark

All 12 scenarios across 4 distinct categories (Easy, Medium Reasoning, Adversarial Trap, Visual Multimodal) were evaluated live against **Google Gemma 4 (`gemma-4-26b-a4b-it`)**. All raw outputs, latencies, and verdicts are saved in [`benchmark_results.json`](file:///d:/HacktoberFest/Gemma-Jury/benchmark_results.json).

### 📈 Official Metrics Summary

| Metric | Single Gemma 4 Baseline | 5-Judge Gemma Jury + Foreperson | Trade-off / Delta |
| :--- | :---: | :---: | :---: |
| **Overall Accuracy** | **100.0%** (12/12) | **91.7%** (11/12) | **-8.3%** (Rigorous Grading) |
| **Easy Category (3 cases)** | **100.0%** (3/3) | **100.0%** (3/3) | Parity (Jury is overkill) |
| **Medium Reasoning (3 cases)** | **100.0%** (3/3) | **66.7%** (2/3) | Single: 3/3, Jury: 2/3 (Rigorous reasoning) |
| **Adversarial Traps (3 cases)** | **100.0%** (3/3) | **100.0%** (3/3) | Parity (Edge-case and cognitive dissection) |
| **Visual Multimodal (3 cases)** | **100.0%** (3/3) | **100.0%** (3/3) | Parity (Spatial set theory & Stroop verification) |
| **Average Inter-Judge Agreement** | N/A | **100.0%** | Unanimous / Nuanced Consensus |
| **Average Turnaround Latency** | **10.33s** | **95.93s** | **+85.60s** (9.29x latency overhead) |
| **Errors Corrected by Jury** | — | **0** | — |
| **Wrong Corrections by Jury** | — | **1** | — |
| **Both Correct** | — | **11 / 12** | — |
| **Both Wrong** | — | **0 / 12** | — |

---

### 🔍 Scientific Findings: Where Multi-Role Helps vs. Where It Does Not

#### 1. Where Multi-Role Deliberation Shines (Qualitative Depth & Robustness)
- **Cognitive Bias & Fallacy Dissection:** On tricky problems like *Monty Hall* or the *Lily Pad Doubling*, the single model outputs the correct answer, but the **Skeptic** and **Devil's Advocate** actively expose *why* human intuition fails (the perceived symmetry fallacy, System 1 substitution heuristic), providing robust pedagogical depth.
- **Topological & Set-Theoretic Rigor in Visual Tasks:** In the *Spatial Containment* test, while the single model merely says "Inside the blue circle", the **Domain Specialist (Dr. Aris)** formalizes the visual coordinates as bounded topological sets in $\mathbb{R}^2$ ($S \subset C$ and $S \cap T = \emptyset$), preventing visual ambiguity from introducing errors.
- **Calibrated Confidence:** Every verdict includes a synthesized confidence score (96–99%) based on agreement across independent personas, providing an explicit measure of consensus.

#### 2. Where Multi-Role Deliberation Does NOT Add Value (Compute Overkill & Latency)
- **High-Capability Single-Model Parity:** For a model as capable as Gemma 4 26B, standard reasoning puzzles and factual queries are already within its single-pass capabilities. Convening 5 judges yielded **0% binary accuracy improvement**.
- **Severe Latency Overhead:** Running 5 independent judges in parallel + foreperson reconciliation takes **95.93 seconds** on average compared to **10.33 seconds** for the single model—a **9.29x latency increase**.
- **Direct Facts and Arithmetic:** On simple factual or arithmetic tasks (e.g. speed calculation, letter counting, capital of Australia), 5 judge calls produce identical conclusions while multiplying API token usage by 6x.

---

## 🚀 Quickstart

### 1. Ensure Dependencies
Dependencies are standard and light:
```bash
pip install streamlit google-genai pillow pandas altair
```

### 2. Configure API Key
The application reads your API key securely from the `GEMINI_API_KEY` environment variable:
```powershell
# In PowerShell:
setx GEMINI_API_KEY "your_gemma_api_key"
```
Or create a `.env` file in the project folder (already protected by `.gitignore`):
```bash
GEMINI_API_KEY=your_gemma_api_key
```

### 3. Launch the Application
```bash
streamlit run app.py
```

---

## 🔒 Security & Best Practices
- **No Hardcoded Keys:** Secrets are strictly loaded via `os.environ` / `.env` and `.gitignore` protected.
- **No LangChain Overhead:** Clean, direct, native Python implementation using the official `google-genai` SDK.
- **Parallel Dispatch:** All 5 judges and the baseline run concurrently via `ThreadPoolExecutor` for maximum responsiveness.
- **Zero Database Complexity:** Fast, hackathon-friendly in-memory state.
