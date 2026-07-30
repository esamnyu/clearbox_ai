import { Component, type ErrorInfo, type ReactNode } from "react";

/**
 * Catches render-time throws so one bad panel degrades instead of blanking
 * the page.
 *
 * The panels below §II each depend on some combination of WebGPU, a 500 MB
 * model load, and a free-tier backend that may be mid-cold-start. Any of those
 * can produce a shape the renderer did not expect. Without a boundary React
 * unmounts the whole tree on the first throw, and a visitor who came to read
 * the bench result gets an empty document with the answer only in the console.
 *
 * Deliberately not a single boundary at the root: wrapping each section means
 * a failure in, say, the attention heatmap leaves the bench table readable.
 */
interface Props {
  children: ReactNode;
  /** Shown in the fallback so the reader knows which part failed. */
  section: string;
}

interface State {
  error: Error | null;
}

export default class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Keep the detail in the console for anyone actually debugging; the UI
    // gets the short version.
    console.error(`[NeuroScope] ${this.props.section} failed to render`, {
      error,
      componentStack: info.componentStack,
    });
  }

  render() {
    const { error } = this.state;
    if (!error) return this.props.children;

    return (
      <div className="border border-vermillion/30 bg-paper/30 px-6 py-5">
        <p className="font-display text-base text-vermillion-light">
          {this.props.section} could not render.
        </p>
        <p className="mt-2 max-w-2xl font-serif text-sm italic leading-relaxed text-slate-400">
          The rest of the page still works — this panel alone is affected.
          Reloading usually clears it; if it persists, the details are in the
          browser console.
        </p>
        <p className="mt-3 font-mono text-xs text-slate-600">{error.message}</p>
        <button
          type="button"
          onClick={() => this.setState({ error: null })}
          className="mt-4 border border-rule px-4 py-2 font-display text-sm text-slate-300 transition-colors hover:border-vermillion-light hover:text-vermillion-light"
        >
          try again
        </button>
      </div>
    );
  }
}
