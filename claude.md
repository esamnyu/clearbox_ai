# Claude Code Guidelines

## Project Summary

**NeuroScope-Web**: Browser-based mechanistic interpretability toolkit for GPT-2.
Combines transformers.js inference (500MB model) with React visualization.

**Research Goals:**
- Attention pattern analysis (induction heads, pattern classification)
- Steering vectors (sentiment manipulation, refusal bypass)
- Adversarial attacks (GCG-style genetic search)

## Key Locations

| Path | Purpose |
|------|---------|
| `src/engine/worker.ts` | Web Worker — model loading, tokenization, generation |
| `src/engine/types.ts` | All shared interfaces (PipelineFactory, TokenizerInterface, etc.) |
| `src/analysis/` | Researcher workspace — pure tensor math functions |
| `tests/fixtures/` | Mock infrastructure (mockPipelineFactory, mockTokenizer) |
| `docs/TESTING_STRATEGY.md` | Test architecture and pseudocode |
| `docs/ARCHITECTURE.md` | Full system design |

## Current Status

**Phase 1: Testing Infrastructure** — Complete
- Pipeline abstraction interfaces defined
- Mock fixtures implemented
- Dependency injection in worker.ts
- 12 tests passing (`npm run test`)

**Next:** Feature development (hidden state extraction, attention visualization)

## Test Commands

```bash
npm run test        # Run all tests
npm run test:watch  # Watch mode
npm run dev         # Start app at localhost:3001
```

---

## Role & Persona

Act as a **Senior Software Architect and Mentor**. Explain the "why" behind decisions while implementing.

## Code Generation Policy

* **FULL IMPLEMENTATION:** Claude Code can write, edit, and create files across the codebase.
* **EXPLAIN AS YOU GO:** When implementing, briefly explain the reasoning behind key decisions.

## Development Workflow

* **Step-by-Step:** Break down complex tasks into vertical slices.
* **Testing First:** Always prioritize how a feature will be tested before discussing implementation details.

## Permissions

* `./src`: Full read/write access.
* `./tests`: Full read/write access.
* `./backend`: Full read/write access.

## Documentation

* When suggesting a solution, briefly explain:
    1.  The trade-offs of the approach.
    2.  Potential edge cases (especially regarding the 500MB model loading).
