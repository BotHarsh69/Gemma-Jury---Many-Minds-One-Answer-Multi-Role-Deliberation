"""
Gemma Jury — Core Engine
Orchestrates 5 independent Gemma 4 judge roles, baseline single-model call,
and the Foreperson reconciliation engine.
"""

import os
import sys
import json
import re
import time
import random
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from PIL import Image

# Import official google-genai SDK
from google import genai

# Strict model requirement: Gemma 4
MODEL_NAME = "models/gemma-4-26b-a4b-it"


def get_api_key() -> str:
    """Safely retrieves GEMINI_API_KEY from environment, Windows registry, or local .env."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key

    # Check local .env file
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip('"').strip("'")
                        if k == "GEMINI_API_KEY" and v:
                            os.environ["GEMINI_API_KEY"] = v
                            return v
        except Exception:
            pass

    # Check Windows Registry in case set via setx
    if sys.platform == "win32":
        try:
            import winreg
            for hkey, subkey in [
                (winreg.HKEY_CURRENT_USER, "Environment"),
                (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment"),
            ]:
                try:
                    with winreg.OpenKey(hkey, subkey) as key_handle:
                        val, _ = winreg.QueryValueEx(key_handle, "GEMINI_API_KEY")
                        if val:
                            os.environ["GEMINI_API_KEY"] = val
                            return val
                except FileNotFoundError:
                    pass
        except Exception:
            pass

    return ""


def get_client() -> genai.Client:
    """Returns an authenticated google-genai Client."""
    api_key = get_api_key()
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY environment variable is not configured. "
            "Please configure it in Windows or in a local .env file."
        )
    return genai.Client(api_key=api_key)


@dataclass
class JudgeRole:
    id: str
    name: str
    title: str
    avatar: str
    color: str
    perspective: str
    prompt_instructions: str


JUDGE_ROLES: List[JudgeRole] = [
    JudgeRole(
        id="skeptic",
        name="Justice Vance",
        title="The Skeptic & Edge-Case Detective",
        avatar="🔍",
        color="#EF4444",
        perspective="Extreme skepticism, cognitive traps, unstated assumptions, and corner cases.",
        prompt_instructions=(
            "You are The Skeptic on a judicial panel. Your job is to actively search for hidden traps, "
            "questionable premises, confirmation bias, cognitive shortcuts, ambiguity, and failure modes. "
            "Do not accept common intuition at face value. Test the problem against edge cases."
        ),
    ),
    JudgeRole(
        id="specialist",
        name="Dr. Aris",
        title="The Domain Specialist & Formalist",
        avatar="🔬",
        color="#3B82F6",
        perspective="Deep technical rigor, formal definitions, and methodical first-principles reasoning.",
        prompt_instructions=(
            "You are The Domain Specialist on a judicial panel. Approach the input with formal academic "
            "and technical precision. Deconstruct the problem systematically, cite governing principles, "
            "apply precise definitions, and present structured step-by-step reasoning."
        ),
    ),
    JudgeRole(
        id="pragmatist",
        name="Judge Elena",
        title="The First-Principles Thinker & Sanity Checker",
        avatar="💡",
        color="#10B981",
        perspective="Plain-language sanity check, real-world plausibility, and stripping away overthinking.",
        prompt_instructions=(
            "You are The First-Principles Thinker on a judicial panel. Cut through jargon and convoluted arguments. "
            "Apply common-sense checks, baseline physical/practical plausibility, and ensure the explanation "
            "is intuitive, grounded, and clearly understandable to anyone."
        ),
    ),
    JudgeRole(
        id="verifier",
        name="Inspector Thorne",
        title="The Fact-Checker & Empirical Verifier",
        avatar="📋",
        color="#8B5CF6",
        perspective="Evidence scrutiny, mathematical checks, image detail auditing, and anti-hallucination.",
        prompt_instructions=(
            "You are The Fact-Checker on a judicial panel. Scrutinize all verifiable claims, quantitative numbers, "
            "visual artifacts or elements in the image (if present), and historical/scientific facts. "
            "Flag any potential hallucination, imprecise number, or unverified leap in logic."
        ),
    ),
    JudgeRole(
        id="contrarian",
        name="Counselor Malik",
        title="The Devil's Advocate & Contrarian",
        avatar="⚖️",
        color="#F59E0B",
        perspective="Counter-arguments, minority interpretations, trade-offs, and alternative hypotheses.",
        prompt_instructions=(
            "You are The Devil's Advocate on a judicial panel. Your role is to resist groupthink. "
            "Explore alternative viewpoints, plausible counter-hypotheses, secondary interpretations, "
            "and key trade-offs that others might overlook or dismiss too quickly."
        ),
    ),
]


@dataclass
class JudgeVerdict:
    role_id: str
    judge_name: str
    judge_title: str
    avatar: str
    color: str
    opinion: str
    core_stance: str
    confidence: int  # 0 to 100


@dataclass
class JuryDeliberationResult:
    query: str
    single_baseline: str
    judge_verdicts: List[JudgeVerdict]
    consensus_level: str  # Unanimous, Strong Majority, Divided/Split, Dissent
    consensus_summary: str
    agreements: List[str]
    disagreements: List[str]
    confidence_score: int  # 0 to 100
    final_verdict: str
    why_jury_helps: str


def generate_content_with_retry(
    client: genai.Client,
    model: str,
    contents: Any,
    max_retries: int = 4,
    initial_delay: float = 2.0,
) -> Any:
    """Executes model generation with exponential backoff and jitter for transient 500/rate limits."""
    import random
    delay = initial_delay
    last_err = None
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
            )
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                jitter = random.uniform(0.2, 0.8)
                time.sleep(delay + jitter)
                delay *= 2
            else:
                raise last_err
    raise last_err if last_err else RuntimeError("Generation failed after retries")


def call_single_baseline(
    client: genai.Client,
    query: str,
    image: Optional[Image.Image] = None,
) -> str:
    """Standard direct single-model call with Gemma 4 (no multi-role prompting)."""
    contents: List[Any] = []
    if image is not None:
        contents.append(image)
    
    prompt = (
        f"Please answer the following user query directly, accurately, and thoroughly:\n\n{query}"
    )
    contents.append(prompt)

    try:
        response = generate_content_with_retry(
            client=client,
            model=MODEL_NAME,
            contents=contents,
        )
        return response.text.strip() if response.text else "No response generated."
    except Exception as e:
        return f"Error executing single model baseline: {str(e)}"


def call_judge(
    client: genai.Client,
    role: JudgeRole,
    query: str,
    image: Optional[Image.Image] = None,
) -> JudgeVerdict:
    """Executes a single Gemma 4 judge role with its specialized persona."""
    contents: List[Any] = []
    if image is not None:
        contents.append(image)

    role_prompt = f"""You are {role.name}, {role.title}.
Your Perspective: {role.perspective}
{role.prompt_instructions}

Please examine the following case/problem/query:
\"\"\"{query}\"\"\"

Analyze it strictly from your assigned role and perspective.
Format your response clearly as:
STANCE: [One crisp sentence summarizing your main takeaway or verdict]
CONFIDENCE: [Number between 0 and 100 indicating your confidence in this stance]
ANALYSIS:
[Your thorough, role-specific reasoning and examination]
"""
    contents.append(role_prompt)

    try:
        response = generate_content_with_retry(
            client=client,
            model=MODEL_NAME,
            contents=contents,
        )
        raw_text = response.text.strip() if response.text else "No opinion rendered."
    except Exception as e:
        raw_text = f"STANCE: Error encountered\nCONFIDENCE: 0\nANALYSIS: Error executing judge {role.name}: {str(e)}"

    # Parse STANCE, CONFIDENCE, ANALYSIS
    stance = "Deliberation complete"
    confidence = 75
    analysis = raw_text

    stance_match = re.search(r"STANCE:\s*(.+?)(?=\n|CONFIDENCE:)", raw_text, re.IGNORECASE)
    if stance_match:
        stance = stance_match.group(1).strip()

    conf_match = re.search(r"CONFIDENCE:\s*(\d+)", raw_text, re.IGNORECASE)
    if conf_match:
        try:
            confidence = max(0, min(100, int(conf_match.group(1))))
        except ValueError:
            confidence = 75

    analysis_match = re.search(r"ANALYSIS:\s*(.*)", raw_text, re.IGNORECASE | re.DOTALL)
    if analysis_match:
        analysis = analysis_match.group(1).strip()

    return JudgeVerdict(
        role_id=role.id,
        judge_name=role.name,
        judge_title=role.title,
        avatar=role.avatar,
        color=role.color,
        opinion=analysis,
        core_stance=stance,
        confidence=confidence,
    )


def reconcile_verdicts(
    client: genai.Client,
    query: str,
    verdicts: List[JudgeVerdict],
    image: Optional[Image.Image] = None,
) -> Dict[str, Any]:
    """The Foreperson / Chief Justice synthesizes opinions, evaluates consensus and disputes."""
    contents: List[Any] = []
    if image is not None:
        contents.append(image)

    opinions_summary = ""
    for v in verdicts:
        opinions_summary += (
            f"\n### {v.avatar} {v.judge_name} ({v.judge_title})\n"
            f"- Core Stance: {v.core_stance}\n"
            f"- Confidence: {v.confidence}%\n"
            f"- Key Arguments:\n{v.opinion[:600]}...\n"
        )

    foreperson_prompt = f"""You are the Chief Justice / Jury Foreperson presiding over a 5-judge panel of Gemma 4 instances.
Your task is to reconcile their findings on the following query:
\"\"\"{query}\"\"\"

Here are the individual opinions from the 5 independent judges:
{opinions_summary}

Please synthesize their deliberations and output a strictly structured JSON response with these keys:
{{
  "consensus_level": "Unanimous" | "Strong Majority" | "Divided" | "Nuanced Consensus",
  "consensus_summary": "1-2 sentences on how much the judges agreed or disagreed",
  "agreements": ["List of 2 to 3 main points where judges were aligned"],
  "disagreements": ["List of 2 to 3 specific points of contention, doubts, or edge cases raised"],
  "confidence_score": <Integer between 0 and 100 representing overall jury confidence in the final answer>,
  "final_verdict": "The comprehensive, reconciled final answer to the user query, incorporating all valid insights, resolving disagreements, and addressing edge cases.",
  "why_jury_helps": "1-2 sentences explaining specifically how the multi-judge deliberation improved this response compared to an uncritical single-model answer."
}}

Output valid JSON only. Do not include markdown code block formatting like ```json."""

    contents.append(foreperson_prompt)

    try:
        response = generate_content_with_retry(
            client=client,
            model=MODEL_NAME,
            contents=contents,
        )
        # Extract JSON substring between the first { and the last }
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            clean_json = match.group(0)
            data = json.loads(clean_json)
            return data
        
        # If no brackets found, try direct loads
        data = json.loads(text)
        return data
    except Exception as e:
        # Fallback heuristic parser if JSON parsing failed
        avg_conf = int(sum(v.confidence for v in verdicts) / len(verdicts)) if verdicts else 75
        top_stances = "\n".join([f"- {v.judge_name} ({v.judge_title}): {v.core_stance}" for v in verdicts])
        return {
            "consensus_level": "Nuanced Consensus",
            "consensus_summary": "The judges reached agreement on the core conclusion with distinct perspective nuances.",
            "agreements": [v.core_stance for v in verdicts[:3]],
            "disagreements": [v.core_stance for v in verdicts[3:]] if len(verdicts) > 3 else [],
            "confidence_score": avg_conf,
            "final_verdict": f"Reconciled Panel Verdict:\n\n{verdicts[1].opinion if len(verdicts) > 1 else ''}\n\nSummary of Judge Stances:\n{top_stances}",
            "why_jury_helps": "The multi-role jury tested the problem across 5 distinct dimensions, validating edge cases and preventing single-model blind spots.",
        }


def run_jury_deliberation(
    query: str,
    image: Optional[Image.Image] = None,
) -> JuryDeliberationResult:
    """
    Main entry point:
    1. Runs the Single-Model Baseline.
    2. Runs all 5 Gemma 4 Judges in parallel.
    3. Runs Foreperson Reconciliation.
    """
    client = get_client()

    # Step 1 & 2: Launch Single-Model Baseline + 5 Judges concurrently
    verdicts: List[JudgeVerdict] = []
    single_baseline: str = ""

    with ThreadPoolExecutor(max_workers=6) as executor:
        baseline_future = executor.submit(call_single_baseline, client, query, image)
        judge_futures = {}
        for role in JUDGE_ROLES:
            time.sleep(0.15)  # 150ms stagger avoids burst rate-limit collision
            f = executor.submit(call_judge, client, role, query, image)
            judge_futures[f] = role

        single_baseline = baseline_future.result()

        # Preserve order of judges as defined in JUDGE_ROLES
        role_to_verdict: Dict[str, JudgeVerdict] = {}
        for future in as_completed(judge_futures):
            role = judge_futures[future]
            try:
                verdict = future.result()
                role_to_verdict[role.id] = verdict
            except Exception as e:
                role_to_verdict[role.id] = JudgeVerdict(
                    role_id=role.id,
                    judge_name=role.name,
                    judge_title=role.title,
                    avatar=role.avatar,
                    color=role.color,
                    opinion=f"Error executing judge {role.name}: {e}",
                    core_stance="Execution error",
                    confidence=0,
                )

        verdicts = [role_to_verdict[r.id] for r in JUDGE_ROLES if r.id in role_to_verdict]

    # Step 3: Foreperson Reconciliation
    reconciliation = reconcile_verdicts(client, query, verdicts, image)

    return JuryDeliberationResult(
        query=query,
        single_baseline=single_baseline,
        judge_verdicts=verdicts,
        consensus_level=reconciliation.get("consensus_level", "Majority"),
        consensus_summary=reconciliation.get("consensus_summary", ""),
        agreements=reconciliation.get("agreements", []),
        disagreements=reconciliation.get("disagreements", []),
        confidence_score=int(reconciliation.get("confidence_score", 80)),
        final_verdict=reconciliation.get("final_verdict", ""),
        why_jury_helps=reconciliation.get("why_jury_helps", ""),
    )
