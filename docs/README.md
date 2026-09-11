# RepoMind Documentation

This directory contains architectural documentation and design decisions.

## Contents

- **`architecture.md`**: Complete system architecture, component responsibilities, data flow, and technology choices
- (Future) **`api.md`**: Public API documentation
- (Future) **`evaluation.md`**: Evaluation methodology and results

## For Developers

Before implementing any new subsystem:
1. Read the relevant section in `architecture.md`
2. Understand the component's single responsibility
3. Check dependencies and integration points
4. Review the data structures flowing in and out

## For Interviewers

The `architecture.md` file explains:
- Why code-aware chunking matters
- Why hybrid retrieval beats vector-only search
- How the ReAct agent loop works
- Design tradeoffs and when to implement vs. use libraries

This project is designed to demonstrate production-quality system design skills.
