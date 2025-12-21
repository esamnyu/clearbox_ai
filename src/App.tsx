/**
 * Session 1: Minimal app to verify model loading and tokenization.
 *
 * CHECKPOINT TEST:
 * 1. Type "Hello world"
 * 2. See tokens: ["Hello", " world"]
 * 3. See IDs: [15496, 995]
 */

import { useEffect, useState } from 'react';
import { useModelStore } from './store/modelStore';

export default function App() {
  const [prompt, setPrompt] = useState('Hello world');
  const { status, loadProgress, tokens, tokenIds, initWorker, loadModel, tokenize } = useModelStore();

  // Initialize worker on mount
  useEffect(() => {
    initWorker();
  }, [initWorker]);

  // Handle model loading
  const handleLoadModel = async () => {
    await loadModel('Xenova/gpt2');
  };

  // Handle tokenization
  const handleTokenize = async () => {
    if (status === 'ready') {
      await tokenize(prompt);
    }
  };

  return (
    <div className="min-h-screen p-8 max-w-4xl mx-auto">
      <h1 className="text-3xl font-bold mb-2">NeuroScope-Web</h1>
      <p className="text-slate-400 mb-8">Session 1: Model Loading & Tokenization</p>

      {/* Model Loading */}
      <section className="mb-8 p-4 bg-slate-900 rounded-lg border border-slate-800">
        <h2 className="text-lg font-semibold mb-4">1. Load Model</h2>

        {status === 'idle' && (
          <button
            onClick={handleLoadModel}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-500 rounded font-medium"
          >
            Load GPT-2 (124M)
          </button>
        )}

        {status === 'loading' && (
          <div className="space-y-2">
            <div className="text-slate-400">Loading model... {loadProgress.toFixed(0)}%</div>
            <div className="h-2 bg-slate-800 rounded overflow-hidden">
              <div
                className="h-full bg-blue-500 transition-all duration-300"
                style={{ width: `${loadProgress}%` }}
              />
            </div>
          </div>
        )}

        {status === 'ready' && (
          <div className="text-green-400">Model loaded successfully</div>
        )}

        {status === 'error' && (
          <div className="text-red-400">Failed to load model</div>
        )}
      </section>

      {/* Tokenization */}
      <section className="mb-8 p-4 bg-slate-900 rounded-lg border border-slate-800">
        <h2 className="text-lg font-semibold mb-4">2. Tokenize Text</h2>

        <div className="space-y-4">
          <input
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="Enter text to tokenize..."
            className="w-full px-4 py-2 bg-slate-800 border border-slate-700 rounded font-mono"
            disabled={status !== 'ready'}
          />

          <button
            onClick={handleTokenize}
            disabled={status !== 'ready'}
            className="px-4 py-2 bg-green-600 hover:bg-green-500 disabled:opacity-50 disabled:cursor-not-allowed rounded font-medium"
          >
            Tokenize
          </button>
        </div>
      </section>

      {/* Token Display */}
      {tokens.length > 0 && (
        <section className="p-4 bg-slate-900 rounded-lg border border-slate-800">
          <h2 className="text-lg font-semibold mb-4">3. Results</h2>

          {/* Tokens */}
          <div className="mb-4">
            <div className="text-sm text-slate-400 mb-2">Tokens ({tokens.length})</div>
            <div className="flex flex-wrap gap-2">
              {tokens.map((token, i) => (
                <span key={i} className="token">
                  {JSON.stringify(token)}
                </span>
              ))}
            </div>
          </div>

          {/* Token IDs */}
          <div>
            <div className="text-sm text-slate-400 mb-2">Token IDs</div>
            <div className="flex flex-wrap gap-2">
              {tokenIds.map((id, i) => (
                <span key={i} className="token text-blue-400">
                  {id}
                </span>
              ))}
            </div>
          </div>
        </section>
      )}

      {/* Instructions */}
      <section className="mt-8 p-4 bg-slate-900/50 rounded-lg border border-slate-800/50 text-slate-400">
        <h3 className="font-semibold text-slate-300 mb-2">Session 1 Checkpoint</h3>
        <ol className="list-decimal list-inside space-y-1 text-sm">
          <li>Click "Load GPT-2" and wait for download (~500MB)</li>
          <li>Type "Hello world" in the input</li>
          <li>Click "Tokenize"</li>
          <li>Verify tokens: ["Hello", " world"]</li>
          <li>Verify IDs: [15496, 995]</li>
        </ol>
      </section>
    </div>
  );
}
