"""Evaluation dataset loading and data models.

This module defines the data structures for evaluation questions, ground truth,
and reference answers, plus utilities for loading JSONL datasets.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any
import json


@dataclass(frozen=True)
class RetrievalGroundTruth:
    """Ground truth for retrieval evaluation.

    Identifies the code chunks/files that are relevant to a question.
    """
    question_id: str
    question: str
    relevant_chunk_ids: List[str] = field(default_factory=list)
    relevant_filepaths: List[str] = field(default_factory=list)
    category: str = "general"
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RetrievalGroundTruth":
        return cls(
            question_id=data["question_id"],
            question=data["question"],
            relevant_chunk_ids=data.get("relevant_chunk_ids", []),
            relevant_filepaths=data.get("relevant_filepaths", []),
            category=data.get("category", "general"),
            difficulty=data.get("difficulty", "medium"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question": self.question,
            "relevant_chunk_ids": self.relevant_chunk_ids,
            "relevant_filepaths": self.relevant_filepaths,
            "category": self.category,
            "difficulty": self.difficulty,
        }


@dataclass(frozen=True)
class AnswerGroundTruth:
    """Ground truth for answer evaluation.

    Contains a reference answer, key facts that must be present,
    and required citation locations.
    """
    question_id: str
    reference_answer: str
    key_facts: List[str] = field(default_factory=list)
    required_citations: List[str] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnswerGroundTruth":
        return cls(
            question_id=data["question_id"],
            reference_answer=data["reference_answer"],
            key_facts=data.get("key_facts", []),
            required_citations=data.get("required_citations", []),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question_id": self.question_id,
            "reference_answer": self.reference_answer,
            "key_facts": self.key_facts,
            "required_citations": self.required_citations,
        }


@dataclass(frozen=True)
class EvaluationQuestion:
    """Combined evaluation question with both retrieval and answer ground truth."""
    retrieval: RetrievalGroundTruth
    answer: Optional[AnswerGroundTruth] = None

    @property
    def question_id(self) -> str:
        return self.retrieval.question_id

    @property
    def question(self) -> str:
        return self.retrieval.question

    @property
    def category(self) -> str:
        return self.retrieval.category

    @property
    def difficulty(self) -> str:
        return self.retrieval.difficulty


def load_retrieval_dataset(path: Path) -> List[RetrievalGroundTruth]:
    """Load retrieval ground truth from a JSONL file.

    Each line must be a JSON object with fields:
    - question_id (str)
    - question (str)
    - relevant_chunk_ids (list[str], optional)
    - relevant_filepaths (list[str], optional)
    - category (str, optional)
    - difficulty (str, optional)

    Args:
        path: Path to JSONL file

    Returns:
        List of RetrievalGroundTruth objects
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(RetrievalGroundTruth.from_dict(data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            except KeyError as e:
                raise ValueError(f"Missing required field {e} on line {line_num}")
    return results


def load_answer_dataset(path: Path) -> List[AnswerGroundTruth]:
    """Load answer ground truth from a JSONL file.

    Each line must be a JSON object with fields:
    - question_id (str)
    - reference_answer (str)
    - key_facts (list[str], optional)
    - required_citations (list[str], optional)

    Args:
        path: Path to JSONL file

    Returns:
        List of AnswerGroundTruth objects
    """
    results = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                results.append(AnswerGroundTruth.from_dict(data))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num}: {e}")
            except KeyError as e:
                raise ValueError(f"Missing required field {e} on line {line_num}")
    return results


def load_combined_dataset(
    retrieval_path: Path,
    answer_path: Optional[Path] = None
) -> List[EvaluationQuestion]:
    """Load and combine retrieval and answer datasets.

    Args:
        retrieval_path: Path to retrieval ground truth JSONL
        answer_path: Optional path to answer ground truth JSONL

    Returns:
        List of EvaluationQuestion objects with both ground truths merged by question_id
    """
    retrieval_items = load_retrieval_dataset(retrieval_path)
    retrieval_map = {item.question_id: item for item in retrieval_items}

    answer_map = {}
    if answer_path and answer_path.exists():
        answer_items = load_answer_dataset(answer_path)
        answer_map = {item.question_id: item for item in answer_items}

    combined = []
    for qid, retrieval_gt in retrieval_map.items():
        combined.append(EvaluationQuestion(
            retrieval=retrieval_gt,
            answer=answer_map.get(qid)
        ))

    return combined


def split_dataset(
    questions: List[EvaluationQuestion],
    dev_ratio: float = 0.3,
    seed: int = 42
) -> tuple[List[EvaluationQuestion], List[EvaluationQuestion]]:
    """Split dataset into dev and test sets.

    Stratified by category to ensure balanced splits.

    Args:
        questions: List of evaluation questions
        dev_ratio: Fraction to allocate to dev set (default 0.3)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (dev_set, test_set)
    """
    import random

    # Group by category
    by_category: Dict[str, List[EvaluationQuestion]] = {}
    for q in questions:
        by_category.setdefault(q.category, []).append(q)

    dev_set = []
    test_set = []

    rng = random.Random(seed)
    for cat, items in by_category.items():
        rng.shuffle(items)
        split_idx = max(1, int(len(items) * dev_ratio))
        dev_set.extend(items[:split_idx])
        test_set.extend(items[split_idx:])

    return dev_set, test_set


def filter_by_category(
    questions: List[EvaluationQuestion],
    categories: List[str]
) -> List[EvaluationQuestion]:
    """Filter questions to only those in the given categories."""
    cat_set = set(categories)
    return [q for q in questions if q.category in cat_set]


def filter_by_difficulty(
    questions: List[EvaluationQuestion],
    difficulties: List[str]
) -> List[EvaluationQuestion]:
    """Filter questions to only those in the given difficulties."""
    diff_set = set(difficulties)
    return [q for q in questions if q.difficulty in diff_set]