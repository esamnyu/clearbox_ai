/**
 * NeuroScope-Web - Session 1: Model Loading & Tokenization
 * ==========================================================
 *
 * This is the main application component for Session 1 of NeuroScope-Web.
 * It provides a simple UI to verify that the model loading and tokenization
 * pipeline is working correctly.
 *
 * WHAT THIS FILE DOES:
 * 1. Initializes the Web Worker on mount (for background model loading)
 * 2. Provides a button to load the GPT-2 model (~500MB download)
 * 3. Shows loading progress with a progress bar
 * 4. Allows text input for tokenization
 * 5. Displays tokens and their IDs for verification
 *
 * CHECKPOINT TEST:
 * 1. Click "Load GPT-2" and wait for download
 * 2. Type "Hello world"
 * 3. Click "Tokenize"
 * 4. Verify tokens: ["Hello", " world"]
 * 5. Verify IDs: [15496, 995]
 *
 * ERROR HANDLING:
 * - If model loading fails, an error message is shown with a retry button
 * - If the worker fails to initialize, the error is displayed
 *
 * @module App
 */

import { useEffect, useState } from 'react';
import { useModelStore } from './store/modelStore';

export default function App() {
  const [prompt, setPrompt] = useState('Hello world');
  const {
    status,
    loadProgress,
    error,
    tokens,
    tokenIds,
    initWorker,
    loadModel,
    tokenize,
    reset,
  } = useModelStore();

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
          <div className="space-y-3">
            <div className="text-red-400 font-medium">Failed to load model</div>
            {error && (
              <div className="text-sm text-red-300 bg-red-900/20 p-3 rounded border border-red-800">
                {error}
              </div>
            )}
            <button
              onClick={() => {
                reset();
                handleLoadModel();
              }}
              className="px-4 py-2 bg-red-600 hover:bg-red-500 rounded font-medium"
            >
              Retry
            </button>
          </div>
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
          <li>Click "Load GPT-2" and wait for download (~500MB) (loading functionality fixed)</li>
          <li>Type "Hello world" in the input</li>
          <li>Click "Tokenize" (tokenization functionality fixed)</li>
          <li>Verify tokens: ["Hello", " world"] (token display fixed)</li>
          <li>Verify IDs: [15496, 995] (token ID casting fixed)</li>
        </ol>
      </section>
    </div>
  );
}
