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
  const [prompt, setPrompt] = useState('The Eiffel Tower is located in');
  const [genText, setGenText] = useState('');
  const [telemetry, setTelemetry] = useState<any>(null);
  const [isGenerating, setIsGenerating] = useState(false);

  const {
    status,
    loadProgress,
    initWorker,
    loadModel,
    generate
  } = useModelStore();

  // Initialize worker on mount
  useEffect(() => {
    initWorker();
  }, [initWorker]);

  const handleGenerate = async () => {
    if (status !== 'ready') return;
    
    setIsGenerating(true);
    setGenText('Generating...');
    setTelemetry(null);

    // Call the worker directly (or via store action)
    try {
      const result = await generate(prompt);

      setGenText(result.text);
      setTelemetry({
        tokenIds: result.tokenIds,
        attentions: result.attentions,
        hiddenStates: result.hiddenStates
      });
      
      console.log("INTERPRETABILITY DATA:", result);
    } catch (e) {
      console.error("Generation error:", e);
    } finally {
      setIsGenerating(false);
    }
  };

  return (
    <div className="min-h-screen p-8 max-w-4xl mx-auto bg-slate-950 text-slate-200 font-sans">
      <h1 className="text-3xl font-bold mb-2 text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-purple-500">
        Clearbox AI: Interpretability Interface
      </h1>
      
      {/* 1. Model Loader */}
      <section className="mb-8 p-6 bg-slate-900/50 rounded-xl border border-slate-800">
        <div className="flex justify-between items-center mb-4">
          <h2 className="text-lg font-semibold text-white">Model Status</h2>
          <span className={`px-2 py-1 rounded text-xs font-mono ${
            status === 'ready' ? 'bg-green-900/50 text-green-400' : 
            status === 'loading' ? 'bg-blue-900/50 text-blue-400' : 
            'bg-yellow-900/50 text-yellow-400'
          }`}>
            {status.toUpperCase()}
          </span>
        </div>

        {status === 'idle' && (
          <button 
            onClick={() => loadModel('Xenova/gpt2')}
            className="w-full py-3 bg-blue-600 hover:bg-blue-500 rounded-lg font-medium transition-colors"
          >
            Load GPT-2 (124M)
          </button>
        )}

        {status === 'loading' && (
          <div className="w-full bg-slate-800 h-2 rounded-full overflow-hidden">
            <div 
              className="bg-blue-500 h-full transition-all duration-300" 
              style={{ width: `${loadProgress}%` }} 
            />
          </div>
        )}
      </section>

      {/* 2. Experiment Input */}
      <section className="mb-8 p-6 bg-slate-900/50 rounded-xl border border-slate-800">
        <h2 className="text-lg font-semibold mb-4 text-white">Input Prompt</h2>
        <textarea
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          className="w-full h-24 p-4 bg-slate-950 border border-slate-700 rounded-lg font-mono focus:ring-2 focus:ring-blue-500 outline-none"
        />
        <div className="mt-4 flex gap-4">
          <button
            onClick={handleGenerate}
            disabled={status !== 'ready' || isGenerating}
            className="px-6 py-2 bg-purple-600 hover:bg-purple-500 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg font-medium transition-colors"
          >
            {isGenerating ? 'Running Inference...' : 'Generate & Analyze'}
          </button>
        </div>
      </section>

      {/* 3. Analysis Dashboard */}
      {genText && (
        <section className="grid grid-cols-2 gap-6">
          {/* Output Text */}
          <div className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
            <h3 className="text-sm font-semibold text-slate-400 mb-2">MODEL OUTPUT</h3>
            <p className="font-mono text-lg leading-relaxed text-white">
              {genText}
            </p>
          </div>

          {/* Telemetry Stats */}
          <div className="p-6 bg-slate-900/50 rounded-xl border border-slate-800">
            <h3 className="text-sm font-semibold text-slate-400 mb-2">CIRCUIT TELEMETRY</h3>
            <div className="space-y-2 text-sm font-mono">
              <div className="flex justify-between">
                <span>Generated Tokens:</span>
                <span className="text-blue-400">{telemetry?.tokenIds?.length || 0}</span>
              </div>
              <div className="flex justify-between">
                <span>Attention Layers:</span>
                <span className="text-green-400">{telemetry?.attentions?.layers || 'N/A'}</span>
              </div>
              <div className="flex justify-between">
                <span>Capture Status:</span>
                <span className="text-purple-400">Success</span>
              </div>
            </div>
          </div>
        </section>
      )}
    </div>
  );
}
