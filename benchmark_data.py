"""
Gemma Jury — 12-Scenario Rigorous Benchmark Test Suite
Covers 4 distinct categories:
- Easy Factual / Direct Arithmetic (3 cases)
- Medium Reasoning & Logic (3 cases)
- Adversarial / Trick / Ambiguity Traps (3 cases)
- Multimodal Visual Reasoning (3 cases)
"""

import os
from typing import List, Optional
from dataclasses import dataclass

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_assets")


@dataclass
class BenchmarkCase:
    id: str
    category: str  # "Easy", "Medium Reasoning", "Adversarial Trap", "Visual Reasoning"
    title: str
    prompt: str
    image_rel_path: Optional[str]
    trap_or_challenge: str
    ground_truth_summary: str
    expected_behavior: str

    @property
    def ground_truth(self) -> str:
        return self.ground_truth_summary

    @property
    def ground_truth_key_points(self) -> List[str]:
        return [self.ground_truth_summary]


BENCHMARK_CASES: List[BenchmarkCase] = [
    # -------------------------------------------------------------
    # CATEGORY 1: EASY (Control / Standard Retrieval & Arithmetic)
    # -------------------------------------------------------------
    BenchmarkCase(
        id="case_1_easy_capital",
        category="Easy",
        title="Capital City & Currency of Australia",
        prompt="What is the capital city of Australia, and what is its official currency? Name the city and the currency clearly.",
        image_rel_path=None,
        trap_or_challenge="Common misconception is Sydney or Melbourne; official capital is Canberra.",
        ground_truth_summary="Canberra, Australian Dollar (AUD)",
        expected_behavior="Both Single and Jury should pass with 100% accuracy. Jury is compute overkill here.",
    ),
    BenchmarkCase(
        id="case_2_easy_speed",
        category="Easy",
        title="Constant Speed Distance Calculation",
        prompt="A car travels at a constant speed of 60 miles per hour for 2 hours and 30 minutes. How many miles does it travel in total? Provide the exact number of miles.",
        image_rel_path=None,
        trap_or_challenge="Converting 2h 30m to 2.5 hours before multiplying (60 * 2.5 = 150).",
        ground_truth_summary="150 miles",
        expected_behavior="Straightforward deterministic arithmetic; both should pass easily.",
    ),
    BenchmarkCase(
        id="case_3_easy_letters",
        category="Easy",
        title="Character Count in 'ALPHABET'",
        prompt="How many letters are in the English word 'ALPHABET'? Count the letters and state the exact number.",
        image_rel_path=None,
        trap_or_challenge="Simple character counting test.",
        ground_truth_summary="8 letters",
        expected_behavior="Both should pass accurately.",
    ),

    # -------------------------------------------------------------
    # CATEGORY 2: MEDIUM REASONING (Multi-Step Logic & Semantics)
    # -------------------------------------------------------------
    BenchmarkCase(
        id="case_4_medium_knights",
        category="Medium Reasoning",
        title="Knights & Knaves Island Puzzle",
        prompt=(
            "On an island, inhabitants are either Knights (who always tell the truth) or Knaves (who always lie). "
            "You meet two inhabitants, A and B.\n"
            "A says: 'Both of us are Knaves.'\n"
            "What is A, and what is B? State explicitly whether A is a Knight or Knave, and whether B is a Knight or Knave."
        ),
        image_rel_path=None,
        trap_or_challenge="Logical deduction: If A were Knight, his claim would be true, which is contradictory. Thus A is Knave. His statement is false, so B must be Knight.",
        ground_truth_summary="A is a Knave; B is a Knight",
        expected_behavior="Requires multi-step propositional logic.",
    ),
    BenchmarkCase(
        id="case_5_medium_sheep",
        category="Medium Reasoning",
        title="The 'All But Nine' Inventory Riddle",
        prompt="A farmer has 17 sheep in a pasture. A storm hits, and all but 9 run away. How many sheep does the farmer have left? State the number clearly.",
        image_rel_path=None,
        trap_or_challenge="Linguistic distraction: rapid intuition tends to subtract 17 - 9 = 8, but 'all but 9' literally means 9 remained.",
        ground_truth_summary="9 sheep remain",
        expected_behavior="Tests syntactic parsing vs hasty arithmetic subtraction.",
    ),
    BenchmarkCase(
        id="case_6_medium_age",
        category="Medium Reasoning",
        title="Relative Age Invariance Trap",
        prompt="When I was 6 years old, my sister was half my age. Now I am 70 years old. How old is my sister today? State her exact age.",
        image_rel_path=None,
        trap_or_challenge="Heuristic error: dividing 70 by 2 = 35. The actual age difference is constant (6 - 3 = 3 years difference, so 70 - 3 = 67).",
        ground_truth_summary="67 years old",
        expected_behavior="Tests whether the model applies constant offset rather than relative ratio.",
    ),

    # -------------------------------------------------------------
    # CATEGORY 3: ADVERSARIAL TRAP (High Single-Model Failure Rate)
    # -------------------------------------------------------------
    BenchmarkCase(
        id="case_7_adversarial_lilypad",
        category="Adversarial Trap",
        title="Exponential Growth Half-Lake Trap",
        prompt=(
            "In a lake, there is a patch of lily pads. Every single day, the patch doubles in size. "
            "If it takes 48 days for the patch to cover the entire lake, how many days does it take for the patch to cover exactly half of the lake? "
            "Give the exact number of days."
        ),
        image_rel_path=None,
        trap_or_challenge="Linear heuristic trap: dividing 48 by 2 = 24 days. In doubling growth, half-capacity is reached on the immediately preceding day (day 47).",
        ground_truth_summary="47 days",
        expected_behavior="Skeptic and Specialist catch the reverse exponential step.",
    ),
    BenchmarkCase(
        id="case_8_adversarial_vacuum",
        category="Adversarial Trap",
        title="Free Fall in Vacuum (False Premise Trap)",
        prompt=(
            "A 10-pound solid steel bowling ball and a 1-pound wooden ball are dropped simultaneously from a height of 50 meters in a complete vacuum. "
            "Which ball hits the ground first, and by how many seconds? Answer directly."
        ),
        image_rel_path=None,
        trap_or_challenge="False premise: assumes one hits first due to weight. In a vacuum, gravitational acceleration is identical ($g$), so they hit at the exact same time (0 seconds difference).",
        ground_truth_summary="They hit at the exact same time (neither hits first; 0 seconds difference)",
        expected_behavior="Tests whether the model rejects the false premise or hallucinate a heavier-ball advantage.",
    ),
    BenchmarkCase(
        id="case_9_adversarial_monty_hall",
        category="Adversarial Trap",
        title="The Monty Hall Probability Dilemma",
        prompt=(
            "In a game show, you choose Door 1 out of 3 doors (one has a car, two have goats). "
            "The host, who knows what is behind each door, opens Door 3 to reveal a goat. "
            "He then asks: 'Do you want to switch to Door 2, or stick with Door 1?'\n"
            "Does switching increase your probability of winning, decrease it, or does it remain 50/50? "
            "State whether you should switch, and state the exact probability of winning if you switch."
        ),
        image_rel_path=None,
        trap_or_challenge="Extremely common human/LLM cognitive intuition trap: believing the remaining two doors have an equal 50/50 chance. In reality, switching yields 2/3 (~66.7%) probability.",
        ground_truth_summary="Switching increases probability to 2/3 (or ~66.7%)",
        expected_behavior="Tests rigorous Bayesian/probabilistic reasoning over superficial symmetry.",
    ),

    # -------------------------------------------------------------
    # CATEGORY 4: VISUAL REASONING (Multimodal Image Tasks)
    # -------------------------------------------------------------
    BenchmarkCase(
        id="case_10_visual_shapes",
        category="Visual Reasoning",
        title="Visual Shape Counting (Red Circles & Blue Squares)",
        prompt="Examine the image carefully. How many red circles and how many blue squares are present in the image? State your answer in the exact format: 'Red circles: X, Blue squares: Y'.",
        image_rel_path="shapes_count.png",
        trap_or_challenge="Accurate optical object detection and color segregation without hallucination.",
        ground_truth_summary="Red circles: 5, Blue squares: 3",
        expected_behavior="Requires precise multimodal spatial counting.",
    ),
    BenchmarkCase(
        id="case_11_visual_stroop",
        category="Visual Reasoning",
        title="Visual Stroop Effect Conflict",
        prompt="Examine the image. What English word is printed in the image, and what is the font color of the text? Format your answer as: 'Word: [word], Font Color: [color]'.",
        image_rel_path="stroop_test.png",
        trap_or_challenge="The word 'GREEN' is rendered in bright RED font color. Tests text OCR vs color perception conflict.",
        ground_truth_summary="Word: GREEN, Font Color: RED",
        expected_behavior="Tests whether the model confuses the semantic word with the visual font color.",
    ),
    BenchmarkCase(
        id="case_12_visual_spatial",
        category="Visual Reasoning",
        title="Spatial Containment Relationship",
        prompt="Look at the image. Is the yellow star inside the blue circle or inside the green triangle? Answer clearly with either 'Inside the blue circle' or 'Inside the green triangle'.",
        image_rel_path="spatial_star.png",
        trap_or_challenge="Visual topological containment: star is inside the circle; triangle is empty.",
        ground_truth_summary="Inside the blue circle",
        expected_behavior="Tests topological/spatial relationship understanding in images.",
    ),
]
