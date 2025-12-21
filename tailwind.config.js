/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ['JetBrains Mono', 'Fira Code', 'monospace'],
      },
      colors: {
        // Custom colors for tensor visualization
        attention: {
          low: '#1e293b',
          mid: '#3b82f6',
          high: '#f59e0b',
        },
      },
    },
  },
  plugins: [],
}
