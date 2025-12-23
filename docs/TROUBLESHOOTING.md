# Troubleshooting Guide

## App Not Running / Model Loading Issues

When the app fails to run or model loading doesn't work, follow this diagnostic workflow:

### Step 1: Check Dependencies
```bash
# Verify node_modules exists
ls node_modules/@xenova/transformers

# If missing, install dependencies
npm install
```

### Step 2: Check for TypeScript Errors
```bash
npm run type-check
```

Common issues:
- Unused variables in placeholder implementations → Prefix with `_`
- Type mismatches with transformers.js → Use proper type imports or assertions

### Step 3: Verify Dev Server Starts
```bash
npm run dev
# Should start at http://localhost:3001
```

### Step 4: Check Browser Console
- Look for Worker initialization errors
- Check for CORS/COEP header issues (required for SharedArrayBuffer)
- Verify model download progress in Network tab

---

## Graceful Fallback Implementation

Key files with error handling:

| File | What it handles |
|------|-----------------|
| `src/store/modelStore.ts` | Worker init errors, `reset()` action for retry |
| `src/App.tsx` | Detailed error display, retry button |
| `src/engine/worker.ts` | Pipeline type safety, progress tracking |

---

## Common Error Messages

### "Worker not initialized"
- The Web Worker failed to start
- Check browser console for script loading errors
- Try refreshing the page

### "Failed to load model"
- Network issue during ~500MB download
- Click "Retry" button to try again
- Check if Hugging Face CDN is accessible

### TypeScript errors on `npm install`
- The `prepare` script runs `type-check`
- Fix type errors before installing succeeds
- Or run `npm install --ignore-scripts` temporarily
