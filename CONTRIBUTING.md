# Contributing to NeuroScope-Web

Thank you for contributing to NeuroScope-Web! This guide covers development workflows, pair programming best practices, and code standards.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Pair Programming Guide](#pair-programming-guide)
- [Code Standards](#code-standards)
- [Testing Guidelines](#testing-guidelines)
- [Git Workflow](#git-workflow)
- [Architecture Decisions](#architecture-decisions)

## Getting Started

### First-Time Setup

```bash
# 1. Clone and install
git clone <your-repo-url> clearbox_ai
cd clearbox_ai
npm install

# 2. Verify setup
npm run lint        # Should pass with no errors
npm test            # Run test suite
npm run dev         # Start dev server

# 3. Open http://localhost:3001 and verify model loading works
```

### Development Environment

**Recommended Tools:**
- **VS Code** with extensions:
  - ESLint
  - TypeScript and JavaScript
  - Tailwind CSS IntelliSense
  - Prettier (optional, not enforced)
- **Chrome DevTools** for debugging Web Workers
- **React DevTools** browser extension

**Required Knowledge:**
- TypeScript (strict mode)
- React Hooks and functional components
- Web Workers and Comlink RPC
- Zustand state management
- Transformers.js basics

## Development Workflow

### Running the Project

```bash
# Development server (hot reload enabled)
npm run dev

# Run tests in watch mode (while developing)
npm run test:watch

# Type-check without building
npx tsc --noEmit

# Lint code
npm run lint
```

### Project Organization

```
src/
├── App.tsx              # UI components (engineers focus here)
├── engine/              # ML infrastructure (shared)
│   ├── types.ts         # Type contracts (shared - review together)
│   └── worker.ts        # Web Worker API (engineers focus)
├── store/               # State management (engineers focus)
│   └── modelStore.ts
└── analysis/            # Tensor operations (researchers focus - not yet created)
    ├── attention.ts
    ├── embeddings.ts
    └── steering/
```

**Separation of Concerns:**
- **Engineers**: UI, state management, worker infrastructure
- **Researchers**: Tensor operations, mechanistic interpretability algorithms
- **Shared**: Type definitions in `src/engine/types.ts`

## Pair Programming Guide

### Roles and Responsibilities

#### Engineer Role

**Focus Areas:**
- React components and UI/UX
- Zustand state management
- Web Worker setup and Comlink RPC
- Vite configuration and build optimization
- Browser compatibility and performance

**Key Files:**
- `src/App.tsx`
- `src/main.tsx`
- `src/store/modelStore.ts`
- `src/engine/worker.ts`
- `vite.config.ts`

#### Researcher Role

**Focus Areas:**
- Tensor operations and numerical analysis
- Attention pattern detection algorithms
- Embedding space exploration (PCA, clustering)
- Steering vector implementations
- Statistical analysis of model outputs

**Key Files:**
- `src/analysis/attention.ts` (to be created)
- `src/analysis/embeddings.ts` (to be created)
- `src/analysis/steering/` (to be created)
- `src/analysis/utils/tensor.ts` (to be created)

#### Shared Responsibilities

**Type Contracts (`src/engine/types.ts`):**
- Review together before changes
- Ensure tensor shapes are correctly typed
- Document expected data formats
- Update both UI and analysis code after changes

### Pair Programming Sessions

#### Session Workflow

1. **Planning (5-10 min)**
   - Review current session goals (e.g., "Session 2: Text Generation")
   - Identify engineer vs researcher tasks
   - Agree on type contracts for new features

2. **Development (45-50 min)**
   - Driver: Types code and shares screen
   - Navigator: Reviews code, suggests improvements
   - Switch roles every 20-25 minutes

3. **Integration (15-20 min)**
   - Merge engineer and researcher code
   - Test end-to-end functionality
   - Run linter and tests
   - Commit together

4. **Retrospective (5 min)**
   - What went well?
   - What needs improvement?
   - Next session planning

#### Communication Best Practices

- **Be explicit about tensor shapes**: "This should be [batch, seq_len, hidden_dim]"
- **Ask questions early**: "Should this return a Promise or sync value?"
- **Share context proactively**: "I'm modifying types.ts, pause your work"
- **Use type-first design**: Define types before implementation

### Live Share / Remote Pairing

**Recommended Tools:**
- VS Code Live Share
- Tuple
- tmux + terminal sharing

**Setup:**
```bash
# Terminal 1 (Driver): Run dev server
npm run dev

# Terminal 2 (Navigator): Run tests in watch mode
npm run test:watch

# Terminal 3 (Shared): Git operations
git status
```

## Code Standards

### TypeScript

**Strict Mode Required:**
```typescript
// tsconfig.json has strict: true
// All code must:
// - Have explicit types for function parameters
// - Avoid 'any' (use 'unknown' if necessary)
// - Handle null/undefined explicitly
// - Use const assertions for literals
```

**Example:**
```typescript
// Good
interface TokenizationResult {
  tokens: string[]
  tokenIds: number[]
  attentionMask: number[]
}

async function tokenize(text: string): Promise<TokenizationResult> {
  // implementation
}

// Bad
async function tokenize(text: any): Promise<any> {
  // implementation
}
```

### React Components

**Functional Components Only:**
```typescript
// Good
const ModelLoader: React.FC = () => {
  const { status, loadModel } = useModelStore()
  return <button onClick={() => loadModel('Xenova/gpt2')}>Load</button>
}

// Bad (class components)
class ModelLoader extends React.Component { /* ... */ }
```

**Hook Rules:**
- Use Zustand for global state
- Use useState for local component state
- Use useEffect sparingly (prefer derived state)
- Custom hooks for reusable logic

### Web Worker Communication

**Use Comlink for Type Safety:**
```typescript
// worker.ts
export const api: ModelWorkerAPI = {
  async loadModel(modelId: string): Promise<void> {
    // implementation
  }
}

Comlink.expose(api)

// main thread
const worker = wrap<ModelWorkerAPI>(new Worker('./worker.ts'))
await worker.loadModel('Xenova/gpt2')  // Type-safe!
```

### Tensor Types

**Always Specify Shapes:**
```typescript
// Good
interface AttentionWeights {
  data: Float32Array
  shape: [number, number, number, number]  // [batch, heads, seq, seq]
  dtype: 'float32'
}

// Bad
interface AttentionWeights {
  data: Float32Array
  shape: number[]
}
```

### File Naming

- React components: `PascalCase.tsx` (e.g., `ModelLoader.tsx`)
- Utilities: `camelCase.ts` (e.g., `tensorUtils.ts`)
- Types: `types.ts` or `*.types.ts`
- Tests: `*.test.ts` or `*.test.tsx`
- Stores: `*Store.ts` (e.g., `modelStore.ts`)

## Testing Guidelines

### Test Structure

```typescript
import { describe, it, expect, vi } from 'vitest'

describe('ModelStore', () => {
  it('should initialize with idle status', () => {
    const store = useModelStore.getState()
    expect(store.status).toBe('idle')
  })

  it('should update status when loading model', async () => {
    const { loadModel } = useModelStore.getState()
    await loadModel('Xenova/gpt2')
    expect(useModelStore.getState().status).toBe('ready')
  })
})
```

### What to Test

**Required:**
- Store state transitions
- Type conversions and tensor shape transformations
- Error handling in worker communication
- Edge cases (empty input, invalid tokens)

**Optional:**
- UI component rendering (focus on logic instead)
- Integration tests (manual testing in browser is fine for now)

### Running Tests

```bash
# Run once
npm test

# Watch mode (during development)
npm run test:watch

# With coverage (future)
npm test -- --coverage
```

## Git Workflow

### Branch Strategy

```bash
# Main branch: stable, deployable code
main

# Feature branches: one per session or feature
feature/session-2-generation
feature/attention-visualization
feature/steering-vectors

# Bugfix branches
fix/worker-initialization
fix/token-display
```

### Commit Messages

**Format:**
```
<type>(<scope>): <subject>

<optional body>
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `refactor`: Code restructuring (no behavior change)
- `docs`: Documentation changes
- `test`: Add or update tests
- `chore`: Build config, dependencies

**Examples:**
```
feat(worker): add attention weights extraction

Implement outputAttentions support in worker.ts to enable
attention pattern visualization in Session 3.

fix(types): correct hidden states tensor shape

Hidden states should be [layers, batch, seq, hidden] not
[batch, seq, hidden]. Updated SerializedTensor interface.

docs(readme): add pair programming section
```

### Pull Request Process

1. **Create feature branch**
   ```bash
   git checkout -b feature/session-2-generation
   ```

2. **Develop and commit incrementally**
   ```bash
   git add src/App.tsx src/store/modelStore.ts
   git commit -m "feat(ui): add text generation input"
   ```

3. **Push and create PR**
   ```bash
   git push origin feature/session-2-generation
   ```

4. **PR Template** (create `.github/pull_request_template.md`):
   ```markdown
   ## Session/Feature
   Session 2: Text Generation

   ## Changes
   - Added GenerationControls component
   - Implemented streaming generation in worker
   - Updated modelStore with generation state

   ## Testing
   - [ ] Manual test: Generated "Hello world" continuation
   - [ ] Verified tokens update in real-time
   - [ ] Tested error handling (empty input)

   ## Screenshots
   [Add UI screenshots if applicable]
   ```

### Code Review Checklist

**For Reviewers:**
- [ ] TypeScript strict mode passes
- [ ] No `any` types (use `unknown` if needed)
- [ ] Tensor shapes documented in types
- [ ] Worker communication is type-safe (Comlink)
- [ ] Tests added for new logic
- [ ] No console.logs left in code
- [ ] Performance considerations (avoid blocking main thread)

## Architecture Decisions

### When to Update Types

**Update `src/engine/types.ts` when:**
- Adding new tensor formats (e.g., attention weights)
- Changing worker API (add/remove methods)
- Adding new store state properties
- Changing data flow between components

**Process:**
1. Discuss with pair programmer
2. Update types.ts first
3. Update worker.ts and stores
4. Update UI components last
5. Run `npx tsc --noEmit` to verify

### Performance Considerations

**Main Thread:**
- Keep UI responsive (60fps target)
- Minimize state updates
- Use React.memo for expensive renders
- Debounce user input (tokenization)

**Web Worker:**
- All model inference must happen in worker
- Use Transferable objects for large tensors
- Avoid blocking operations in worker
- Batch tensor operations when possible

### When to Add Dependencies

**Ask first:**
- Will this add >100KB to bundle?
- Is there a lighter alternative?
- Can we implement it ourselves easily?

**Pre-approved:**
- Transformers.js ecosystem packages
- Radix UI components
- Lodash (individual imports only: `lodash.debounce`)
- D3.js (for visualizations in later sessions)

## Questions?

For architecture questions, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

For ML research workflows, see [docs/RESEARCHER_GUIDE.md](docs/RESEARCHER_GUIDE.md).

For bugs or feature requests, open a GitHub issue.

Happy coding!
