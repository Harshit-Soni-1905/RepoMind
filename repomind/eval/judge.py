"""Optional LLM-as-a-judge for answer quality evaluation.

Uses Google Gemini free-tier to evaluate:
- Faithfulness: Answer grounded in provided context
- Correctness: Answer is factually correct per actual code
- Completeness: Answer addresses all aspects of question

This is OPTIONAL - core benchmarks use deterministic metrics only.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import os
from pathlib import Path


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation."""
    faithfulness: int  # 1-5
    correctness: int   # 1-5
    completeness: int  # 1-5
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @property
    def average_score(self) -> float:
        return (self.faithfulness + self.correctness + self.completeness) / 3.0


class LLMJudge:
    """LLM-based evaluator using Google Gemini.

    Requires GOOGLE_API_KEY environment variable.
    """

    JUDGE_PROMPT = """You are an expert code evaluator. Given a QUESTION, a GENERATED ANSWER, and the SOURCE CODE CHUNKS provided as context to the agent, evaluate the answer on three dimensions:

1. FAITHFULNESS (1-5): Does the answer only claim what is supported by the provided source chunks?
   - 1 = Hallucinates extensively, makes claims not in sources
   - 3 = Some claims not grounded in sources
   - 5 = Fully grounded, every claim traceable to provided chunks

2. CORRECTNESS (1-5): Is the answer factually correct with respect to the actual repository code?
   - 1 = Factually incorrect
   - 3 = Partially correct, some errors
   - 5 = Fully correct per the actual codebase

3. COMPLETENESS (1-5): Does the answer address all aspects of the question?
   - 1 = Misses the main point entirely
   - 3 = Addresses part of the question
   - 5 = Thoroughly answers all parts

IMPORTANT: You only have access to the PROVIDED CHUNKS as context. You do NOT have access to the full repository. Evaluate based on what the agent was given.

Return ONLY a JSON object with this exact format:
{{
  "faithfulness": <1-5>,
  "correctness": <1-5>,
  "completeness": <1-5>,
  "reasoning": "<brief explanation for each score>"
}}

QUESTION: {question}

GENERATED ANSWER: {answer}

SOURCE CODE CHUNKS PROVIDED TO AGENT:
{context_chunks}

"""

    def __init__(self, model_name: str = "gemini-1.5-flash"):
        """Initialize judge with model name.

        Args:
            model_name: Gemini model to use for judging
        """
        self.model_name = model_name
        self._client = None

    def _get_client(self):
        """Lazy initialization of Gemini client."""
        if self._client is None:
            try:
                from google import genai
                api_key = os.environ.get("GOOGLE_API_KEY")
                if not api_key:
                    raise ValueError("GOOGLE_API_KEY environment variable not set")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise RuntimeError("google-genai package not installed. Install with: pip install google-genai")
        return self._client

    def is_available(self) -> bool:
        """Check if judge can be used (API key present, package installed)."""
        try:
            self._get_client()
            return True
        except (ValueError, RuntimeError):
            return False

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: List[Dict[str, Any]],
    ) -> Optional[JudgeResult]:
        """Evaluate an answer using the LLM judge.

        Args:
            question: The original question
            answer: Generated answer to evaluate
            context_chunks: List of source code chunks provided as context

        Returns:
            JudgeResult or None if evaluation failed
        """
        if not self.is_available():
            return None

        # Format context chunks for prompt
        context_str = self._format_context(context_chunks)

        prompt = self.JUDGE_PROMPT.format(
            question=question,
            answer=answer,
            context_chunks=context_str,
        )

        try:
            client = self._get_client()
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
            )

            # Parse JSON response
            text = response.text.strip()
            # Handle potential markdown code fences
            if text.startswith("```json"):
                text = text[7:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

            result = json.loads(text)

            return JudgeResult(
                faithfulness=int(result.get("faithfulness", 0)),
                correctness=int(result.get("correctness", 0)),
                completeness=int(result.get("completeness", 0)),
                reasoning=result.get("reasoning", ""),
            )

        except (json.JSONDecodeError, KeyError, ValueError, AttributeError) as e:
            # Log error and return None
            return None

    def _format_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks for judge prompt."""
        if not chunks:
            return "(No context chunks provided)"

        parts = []
        for i, chunk in enumerate(chunks, 1):
            filepath = chunk.get("filepath", "unknown")
            symbol = chunk.get("symbol_name", "unknown")
            start = chunk.get("start_line", 0)
            end = chunk.get("end_line", 0)
            source = chunk.get("source_code", "")

            parts.append(f"--- Chunk {i} ---")
            parts.append(f"File: {filepath}")
            parts.append(f"Symbol: {symbol}")
            parts.append(f"Lines: {start}-{end}")
            if source:
                parts.append(f"Code:\n{source}")
            parts.append("")

        return "\n".join(parts)


def evaluate_with_judge(
    questions: List[Dict[str, Any]],
    answers: List[Dict[str, Any]],
    context_map: Dict[str, List[Dict[str, Any]]],
    model_name: str = "gemini-1.5-flash"
) -> List[Dict[str, Any]]:
    """Evaluate a batch of QA pairs with the LLM judge.

    Args:
        questions: List of question objects with question_id, question
        answers: List of answer objects with question_id, answer
        context_map: Mapping question_id -> list of context chunks
        model_name: Gemini model name

    Returns:
        List of evaluation records with judge scores
    """
    judge = LLMJudge(model_name=model_name)

    if not judge.is_available():
        return []

    results = []
    for q_obj in questions:
        qid = q_obj.get("question_id")
        question_text = q_obj.get("question", "")

        # Find matching answer
        ans_obj = next((a for a in answers if a.get("question_id") == qid), None)
        if not ans_obj:
            continue

        answer_text = ans_obj.get("answer", "")
        context = context_map.get(qid, [])

        judge_result = judge.evaluate(question_text, answer_text, context)

        if judge_result:
            results.append({
                "question_id": qid,
                "faithfulness": judge_result.faithfulness,
                "correctness": judge_result.correctness,
                "completeness": judge_result.completeness,
                "average_score": judge_result.average_score,
                "reasoning": judge_result.reasoning,
            })

    return results