# Evaluation Data

This directory will contain evaluation datasets for measuring RepoMind's performance.

## Structure

Evaluation datasets will be added during Stage 9 (Evaluation) and will include:

- **Ground truth Q&A pairs**: Questions about codebases with known correct answers
- **Relevance judgments**: Which code chunks are relevant to each question
- **Test repositories**: Small Python repositories with known structure

## Dataset Format

(To be defined during Stage 9 implementation)

Example structure:
```json
{
  "repo": "path/to/test/repo",
  "questions": [
    {
      "id": "q1",
      "question": "How does the authentication system work?",
      "relevant_chunks": ["auth/login.py:15-45", "models/user.py:10-30"],
      "expected_answer": "The authentication system uses..."
    }
  ]
}
```

## Purpose

Evaluation data allows us to:
1. Compare retrieval quality: vector-only vs. hybrid
2. Measure precision@k, recall@k, mean reciprocal rank
3. Assess answer correctness and faithfulness
4. Track improvements as we iterate on the system

---

*This directory is currently empty. Evaluation datasets will be added in Stage 9.*
