# Testing Strategy for `worker.ts`

> **Author:** Architecture Review
> **Status:** Draft
> **Priority Order:** loadModel → tokenize → getStatus → unload

---

## Overview

This document outlines the testing architecture for `src/engine/worker.ts`, the Web Worker responsible for GPT-2 model inference. The strategy addresses the core constraint: **we cannot load the 500MB production model for every test case**.

---

## The 500MB Problem: Three Solutions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODEL LOADING TEST STRATEGY                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   SOLUTION A: Mock Pipeline (Unit Tests)                                    │
│   ────────────────────────────────────────                                  │
│   • Zero network calls                                                      │
│   • Deterministic outputs (critical for adversarial research)               │
│   • Tests logic, not transformers.js                                        │
│   • Run time: <100ms                                                        │
│                                                                             │
│   SOLUTION B: Tiny Real Model (Integration Tests)                           │
│   ────────────────────────────────────────────────                          │
│   • Use: Xenova/gpt2 with quantized weights (ONNX int8)                     │
│   • Or: A custom-trained 1-layer transformer (~2MB)                         │
│   • Tests real tokenizer behavior                                           │
│   • Run time: 2-5 seconds (with browser cache)                              │
│                                                                             │
│   SOLUTION C: Cached Full Model (Smoke Tests, Optional)                     │
│   ─────────────────────────────────────────────────────                     │
│   • Run manually before release                                             │
│   • Use HuggingFace cache to avoid re-download                              │
│   • CI: Skip by default, run nightly                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Recommendation:** Use **Solution A** for all logic tests, **Solution B** for tokenizer correctness (required for adversarial research).

---

## Architectural Change: Dependency Injection

The current `worker.ts` has a hard dependency on `pipeline` from transformers.js. To test without the 500MB model, we need **dependency injection**.

### Interface Definitions

```typescript
// types.ts - Add these interfaces

/**
 * Abstraction over the transformers.js pipeline.
 * Allows injection of mocks for testing.
 */
interface TokenizerInterface {
  (text: string, options: TokenizerOptions): Promise<EncodedTokens>
  decode(ids: number[], options?: DecodeOptions): string
}

interface EncodedTokens {
  input_ids: { data: BigInt64Array | number[] }
  attention_mask?: { data: BigInt64Array | number[] }
}

interface PipelineInterface {
  tokenizer: TokenizerInterface
  (prompt: string, options: GenerationConfig): Promise<GenerationOutput[]>
}

/**
 * Factory that creates pipelines.
 * Production: calls transformers.js
 * Testing: returns mock or tiny model
 */
interface PipelineFactory {
  create(
    task: 'text-generation',
    modelId: ModelId,
    config: PipelineConfig
  ): Promise<PipelineInterface>
}

interface PipelineConfig {
  dtype: string
  device: string
  progress_callback?: (progress: LoadProgress) => void
}
```

### Why This Matters for Adversarial Research

When testing adversarial inputs (Unicode edge cases, token boundary attacks, prompt injections), you need:
1. **Deterministic mock** — to verify detection logic is correct
2. **Real tokenizer** — to verify actual model behavior

The interface lets you swap between them.

---

## Priority 1: Testing `loadModel`

### Behaviors to Test

```
BEHAVIOR 1: State transitions correctly
    idle → loading → ready
    idle → loading → error (on failure)

BEHAVIOR 2: Progress callback fires with correct shape
    { status: 'downloading' | 'loading', progress: 0-100, file?: string }

BEHAVIOR 3: Idempotency
    Loading same model twice → no-op, stays ready

BEHAVIOR 4: Model switching
    Load model A → ready
    Load model B → should unload A first? (design decision)

BEHAVIOR 5: Error recovery
    Load fails → error state
    Load again → should work
```

### Test Structure Pseudocode

```typescript
describe('loadModel', () => {

  let worker: ModelWorkerAPI
  let mockFactory: MockPipelineFactory

  beforeEach(() => {
    mockFactory = createMockPipelineFactory()
    worker = createTestableWorker(mockFactory)
  })

  // ─────────────────────────────────────────────────────────
  // BEHAVIOR 1: State Transitions
  // ─────────────────────────────────────────────────────────

  describe('state transitions', () => {

    test('transitions from idle to loading to ready', async () => {
      // GIVEN: Worker is in idle state
      ASSERT (await worker.getStatus()).status === 'idle'

      // WHEN: loadModel is called (but not awaited)
      const loadPromise = worker.loadModel('Xenova/gpt2')

      // THEN: Status should be 'loading'
      ASSERT (await worker.getStatus()).status === 'loading'

      // WHEN: Load completes
      await loadPromise

      // THEN: Status should be 'ready'
      const finalStatus = await worker.getStatus()
      ASSERT finalStatus.status === 'ready'
      ASSERT finalStatus.modelId === 'Xenova/gpt2'
    })

    test('transitions to error on failure', async () => {
      // GIVEN: Factory configured to fail
      mockFactory.setShouldFail(true, new Error('Network timeout'))

      // WHEN: loadModel is called
      await EXPECT_THROWS(
        worker.loadModel('Xenova/gpt2'),
        'Network timeout'
      )

      // THEN: Status should be 'error', model cleared
      const status = await worker.getStatus()
      ASSERT status.status === 'error'
      ASSERT status.modelId === null
    })
  })

  // ─────────────────────────────────────────────────────────
  // BEHAVIOR 2: Progress Callbacks
  // ─────────────────────────────────────────────────────────

  describe('progress reporting', () => {

    test('fires progress callbacks with correct shape', async () => {
      const progressEvents: LoadProgress[] = []

      await worker.loadModel('Xenova/gpt2', (progress) => {
        progressEvents.push(progress)
      })

      ASSERT progressEvents.length > 0

      FOR each event IN progressEvents:
        ASSERT event.status IN ['downloading', 'loading', 'ready']
        IF event.progress !== undefined:
          ASSERT 0 <= event.progress <= 100
    })
  })

  // ─────────────────────────────────────────────────────────
  // BEHAVIOR 3: Idempotency
  // ─────────────────────────────────────────────────────────

  describe('idempotency', () => {

    test('loading same model twice is a no-op', async () => {
      await worker.loadModel('Xenova/gpt2')
      const callCount = mockFactory.getCreateCallCount()

      await worker.loadModel('Xenova/gpt2')

      ASSERT mockFactory.getCreateCallCount() === callCount
      ASSERT (await worker.getStatus()).status === 'ready'
    })
  })
})
```

### Mock Factory Interface

```typescript
interface MockPipelineFactory extends PipelineFactory {
  // Test controls
  setShouldFail(fail: boolean, error?: Error): void
  setLoadDelay(milliseconds: number): void
  getCreateCallCount(): number
  getLastCreateCall(): { task: string; modelId: string; config: object }
}

FUNCTION createMockPipelineFactory(): MockPipelineFactory {
  LET shouldFail = false
  LET failError = new Error('Mock failure')
  LET loadDelay = 0
  LET createCallCount = 0
  LET lastCreateCall = null

  RETURN {
    async create(task, modelId, config) {
      createCallCount++
      lastCreateCall = { task, modelId, config }

      IF loadDelay > 0:
        await delay(loadDelay)

      IF config.progress_callback:
        config.progress_callback({ status: 'downloading', progress: 50 })
        config.progress_callback({ status: 'loading', progress: 100 })

      IF shouldFail:
        THROW failError

      RETURN createMockPipeline()
    },

    setShouldFail(fail, error) { shouldFail = fail; failError = error },
    setLoadDelay(ms) { loadDelay = ms },
    getCreateCallCount() { return createCallCount },
    getLastCreateCall() { return lastCreateCall }
  }
}
```

---

## Priority 2: Testing `tokenize`

### Test Categories

```
CATEGORY 1: Basic Correctness
    Input → Expected tokens (with known model)

CATEGORY 2: Invariants (Must Always Hold)
    • len(tokens) === len(tokenIds) === len(attentionMask)
    • All tokenIds are valid vocabulary indices (0 to 50256)
    • Decode(Encode(text)) ≈ text (roundtrip)

CATEGORY 3: Edge Cases (Future Adversarial Surface)
    • Empty string
    • Single character
    • Max length input (1024 tokens)
    • Unicode: emoji, RTL text, zero-width chars
    • Newlines and whitespace
    • Special tokens if any

CATEGORY 4: Error Conditions
    • Tokenize before model loaded → throws
```

### Test Pseudocode

```typescript
describe('tokenize', () => {

  // ─────────────────────────────────────────────────────────
  // CATEGORY 1: Basic Correctness (Unit Tests with Mock)
  // ─────────────────────────────────────────────────────────

  describe('with mock tokenizer', () => {

    test('returns correct structure', async () => {
      const mockTokenizer = createMockTokenizer({
        'Hello': { ids: [15496], tokens: ['Hello'] },
      })
      const worker = createTestableWorker(mockTokenizer)
      await worker.loadModel('Xenova/gpt2')

      const result = await worker.tokenize('Hello')

      ASSERT result.tokens === ['Hello']
      ASSERT result.tokenIds === [15496]
      ASSERT result.attentionMask === [1]
    })

    test('throws when no model loaded', async () => {
      const worker = createTestableWorker()

      await EXPECT_THROWS(
        worker.tokenize('Hello'),
        'No model loaded'
      )
    })
  })

  // ─────────────────────────────────────────────────────────
  // CATEGORY 2: Invariants (Integration Tests)
  // ─────────────────────────────────────────────────────────

  describe('invariants (integration)', () => {

    beforeAll(async () => {
      await worker.loadModel('Xenova/gpt2')
    }, LONG_TIMEOUT)

    const TEST_INPUTS = [
      'Hello',
      'Hello, world!',
      'The quick brown fox jumps over the lazy dog.',
      'GPT-2 tokenization test',
      '   leading spaces',
      'trailing spaces   ',
      'multiple   spaces   between',
    ]

    test.each(TEST_INPUTS)('length invariant for: %s', async (input) => {
      const result = await worker.tokenize(input)

      ASSERT result.tokens.length === result.tokenIds.length
      ASSERT result.tokens.length === result.attentionMask.length
    })

    test.each(TEST_INPUTS)('valid token IDs for: %s', async (input) => {
      const result = await worker.tokenize(input)

      const VOCAB_SIZE = 50257
      FOR each id IN result.tokenIds:
        ASSERT id >= 0
        ASSERT id < VOCAB_SIZE
    })

    test.each(TEST_INPUTS)('roundtrip approximate equality: %s', async (input) => {
      const result = await worker.tokenize(input)

      const reconstructed = result.tokens.join('')
      const normalizedInput = normalizeWhitespace(input)
      const normalizedOutput = normalizeWhitespace(reconstructed)

      ASSERT normalizedInput === normalizedOutput
    })

    test('attention mask is all ones (no padding)', async () => {
      const result = await worker.tokenize('Any text here')

      FOR each mask IN result.attentionMask:
        ASSERT mask === 1
    })
  })

  // ─────────────────────────────────────────────────────────
  // CATEGORY 3: Edge Cases (Critical for Adversarial Research)
  // ─────────────────────────────────────────────────────────

  describe('edge cases', () => {

    test('empty string', async () => {
      const result = await worker.tokenize('')
      // DECISION: Document expected behavior
      ASSERT result.tokens.length === 0
    })

    test('single character', async () => {
      const result = await worker.tokenize('a')
      ASSERT result.tokens.length >= 1
    })

    test('unicode emoji', async () => {
      const result = await worker.tokenize('Hello 👋 World')
      ASSERT result.tokens.length > 0
      ASSERT result.tokenIds.every(id => id >= 0 && id < 50257)
    })

    test('newline handling', async () => {
      const result = await worker.tokenize('Line 1\nLine 2')
      ASSERT result.tokens.some(t => t.includes('\n') || t === 'Ċ')
    })

    test('very long input (approaching context limit)', async () => {
      const longInput = 'word '.repeat(1000)
      const result = await worker.tokenize(longInput)
      ASSERT result.tokens.length > 500
    })
  })
})
```

### Mock Tokenizer Interface

```typescript
interface TokenMapping {
  [input: string]: {
    ids: number[]
    tokens: string[]
  }
}

FUNCTION createMockTokenizer(mappings: TokenMapping): TokenizerInterface {
  RETURN {
    async (text, options) {
      IF text IN mappings:
        const mapped = mappings[text]
        RETURN {
          input_ids: { data: BigInt64Array.from(mapped.ids.map(BigInt)) },
          attention_mask: { data: BigInt64Array.from(mapped.ids.map(() => 1n)) }
        }
      ELSE:
        const words = text.split(' ')
        const ids = words.map((_, i) => i + 1000)
        RETURN {
          input_ids: { data: BigInt64Array.from(ids.map(BigInt)) },
          attention_mask: { data: BigInt64Array.from(ids.map(() => 1n)) }
        }
    },

    decode(ids, options) {
      RETURN ids.map(id => `[token-${id}]`).join('')
    }
  }
}
```

---

## Priority 3 & 4: Testing `getStatus` and `unloadModel`

### State Machine

```
                    ┌─────────────────────────────────────────┐
                    │                                         │
                    ▼                                         │
    ┌───────┐  loadModel()  ┌─────────┐  success  ┌───────┐  │
    │ idle  │ ────────────► │ loading │ ────────► │ ready │  │
    └───────┘               └─────────┘           └───────┘  │
        ▲                        │                    │       │
        │                        │ error              │       │
        │                        ▼                    │       │
        │                   ┌─────────┐               │       │
        │                   │  error  │               │       │
        │                   └─────────┘               │       │
        │                        │                    │       │
        │                        │ unloadModel()      │       │
        └────────────────────────┴────────────────────┘       │
                                                              │
                         unloadModel() ───────────────────────┘
```

### Test Pseudocode

```typescript
describe('getStatus', () => {

  test('returns idle on fresh worker', async () => {
    const worker = createTestableWorker()
    const status = await worker.getStatus()

    ASSERT status.status === 'idle'
    ASSERT status.modelId === null
  })

  test('returns ready with modelId after load', async () => {
    const worker = createTestableWorker(mockFactory)
    await worker.loadModel('Xenova/gpt2')

    const status = await worker.getStatus()
    ASSERT status.status === 'ready'
    ASSERT status.modelId === 'Xenova/gpt2'
  })

  test('status reflects loading state mid-load', async () => {
    mockFactory.setLoadDelay(100)

    const loadPromise = worker.loadModel('Xenova/gpt2')

    const midStatus = await worker.getStatus()
    ASSERT midStatus.status === 'loading'

    await loadPromise
  })
})

describe('unloadModel', () => {

  test('transitions from ready to idle', async () => {
    await worker.loadModel('Xenova/gpt2')
    ASSERT (await worker.getStatus()).status === 'ready'

    await worker.unloadModel()

    const status = await worker.getStatus()
    ASSERT status.status === 'idle'
    ASSERT status.modelId === null
  })

  test('unload when idle is a no-op', async () => {
    ASSERT (await worker.getStatus()).status === 'idle'

    await worker.unloadModel()

    ASSERT (await worker.getStatus()).status === 'idle'
  })

  test('tokenize fails after unload', async () => {
    await worker.loadModel('Xenova/gpt2')
    await worker.unloadModel()

    await EXPECT_THROWS(
      worker.tokenize('test'),
      'No model loaded'
    )
  })

  test('can reload after unload', async () => {
    await worker.loadModel('Xenova/gpt2')
    await worker.unloadModel()

    await worker.loadModel('Xenova/gpt2')

    ASSERT (await worker.getStatus()).status === 'ready'
    const result = await worker.tokenize('test')
    ASSERT result.tokens.length > 0
  })
})
```

---

## File Structure

```
tests/
├── setup.ts                         # Vitest config, global mocks
│
├── fixtures/
│   ├── mockPipelineFactory.ts       # Mock PipelineFactory implementation
│   ├── mockTokenizer.ts             # Mock TokenizerInterface
│   ├── testCases.ts                 # Shared test input data
│   └── constants.ts                 # GPT2_VOCAB_SIZE, timeouts, etc.
│
├── unit/
│   ├── worker.loadModel.test.ts     # Priority 1
│   ├── worker.tokenize.test.ts      # Priority 2
│   ├── worker.getStatus.test.ts     # Priority 3
│   └── worker.unload.test.ts        # Priority 4
│
├── integration/
│   ├── tokenizer.invariants.test.ts # Real tokenizer, invariant checks
│   └── tokenizer.edgeCases.test.ts  # Unicode, long inputs, etc.
│
└── e2e/                              # Optional, manual runs
    └── fullModel.smoke.test.ts

src/engine/
├── worker.ts                         # Current (untouched initially)
├── workerFactory.ts                  # NEW: Factory for testability
├── pipelineFactory.ts                # NEW: Wraps transformers.js
├── pipelineFactory.interface.ts      # NEW: Interfaces for DI
└── types.ts                          # Existing (add new interfaces)
```

---

## Implementation Steps

### Step 1: Create Interface Layer
- Add `PipelineFactory`, `PipelineInterface`, `TokenizerInterface` to types
- These are just type definitions, no behavior change yet

### Step 2: Create Mock Fixtures
- Implement `mockPipelineFactory.ts`
- Implement `mockTokenizer.ts`
- Create `constants.ts` with vocab size, timeouts

### Step 3: Create Testable Worker Factory
- Create `workerFactory.ts` that accepts `PipelineFactory`
- This is a thin wrapper, not a rewrite

### Step 4: Write `loadModel` Tests (Priority 1)
- State transitions
- Progress callbacks
- Error handling
- Idempotency

### Step 5: Write `tokenize` Tests (Priority 2)
- Structure validation with mocks
- Invariant tests (can use real model here if needed)
- Edge case documentation

### Step 6: Write `getStatus` / `unload` Tests (Priority 3 & 4)
- State machine verification
- Memory cleanup verification (best effort)

### Step 7: Integration Test Setup
- Configure Vitest for longer timeouts
- Add tiny model loading test
- Set up CI caching for model files

---

## Edge Cases for Adversarial Research

| Category | Input | Expected Behavior | Why It Matters |
|----------|-------|-------------------|----------------|
| **Boundary** | Empty string `""` | Returns empty arrays? Throws? | Prompt injection might rely on empty handling |
| **Unicode** | Zero-width joiner `\u200D` | Tokenizes as byte? Ignored? | Invisible character attacks |
| **Unicode** | RTL override `\u202E` | Preserved in token? | Display manipulation attacks |
| **Length** | 1025+ tokens | Truncate? Throw? | Context window overflow attacks |
| **Special** | `<\|endoftext\|>` literal | Single token ID 50256? | Premature termination injection |

---

## Trade-offs to Decide

| Decision | Option A | Option B |
|----------|----------|----------|
| **Refactor now or wrap?** | Refactor `worker.ts` for DI (cleaner) | Create test wrapper that patches globals (faster) |
| **Tiny model choice** | Use existing HF model (less work) | Train 1MB custom model (full control) |
| **Worker execution** | Test in Node (fast, less realistic) | Test in browser via Playwright (slow, accurate) |

---

## Framework Choice: Vitest

Already configured in `package.json`. Key advantages:

| Criterion | Custom Runner | Vitest |
|-----------|---------------|--------|
| Setup Cost | High | Already configured |
| Worker Support | Manual | Built-in |
| Async/Await | Manual | Native support |
| Mocking | Build from scratch | `vi.mock()` built-in |
| Watch Mode | DIY | `npm run test:watch` |
| CI Integration | Custom | Works out of box |

**Note:** Configure per-test timeouts for integration tests that load real models.
