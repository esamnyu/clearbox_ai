/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        display: ["Fraunces", "ui-serif", "Georgia", "serif"],
        serif: ["Newsreader", "ui-serif", "Georgia", "serif"],
      },
      colors: {
        attention: {
          low: "#1e293b",
          mid: "#3b82f6",
          high: "#f59e0b",
        },
        ink: "#0d0e14",
        paper: "#13141c",
        rule: "#23252f",
        graphite: "#e9eaf0",
        vermillion: {
          DEFAULT: "#bd4931",
          light: "#e9876f",
          muted: "#6d2a1d",
        },
        cerulean: {
          DEFAULT: "#38749c",
          light: "#6aa3c8",
          muted: "#1f4257",
        },
        background: "hsl(var(--background))",
        foreground: "hsl(var(--foreground))",
        card: {
          DEFAULT: "hsl(var(--card))",
          foreground: "hsl(var(--card-foreground))",
        },
        popover: {
          DEFAULT: "hsl(var(--popover))",
          foreground: "hsl(var(--popover-foreground))",
        },
        primary: {
          DEFAULT: "hsl(var(--primary))",
          foreground: "hsl(var(--primary-foreground))",
        },
        secondary: {
          DEFAULT: "hsl(var(--secondary))",
          foreground: "hsl(var(--secondary-foreground))",
        },
        muted: {
          DEFAULT: "hsl(var(--muted))",
          foreground: "hsl(var(--muted-foreground))",
        },
        accent: {
          DEFAULT: "hsl(var(--accent))",
          foreground: "hsl(var(--accent-foreground))",
        },
        destructive: {
          DEFAULT: "hsl(var(--destructive))",
          foreground: "hsl(var(--destructive-foreground))",
        },
        border: "hsl(var(--border))",
        input: "hsl(var(--input))",
        ring: "hsl(var(--ring))",
        chart: {
          1: "hsl(var(--chart-1))",
          2: "hsl(var(--chart-2))",
          3: "hsl(var(--chart-3))",
          4: "hsl(var(--chart-4))",
          5: "hsl(var(--chart-5))",
        },
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
      keyframes: {
        reveal: {
          "0%": { opacity: "0", transform: "translateY(6px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
        "rule-grow": {
          "0%": { transform: "scaleX(0)", opacity: "0" },
          "100%": { transform: "scaleX(1)", opacity: "1" },
        },
        "rule-pulse": {
          "0%, 100%": { opacity: "0.35" },
          "50%": { opacity: "0.85" },
        },
        "math-glow": {
          "0%, 100%": { textShadow: "0 0 0 rgba(189,73,49,0)" },
          "50%": { textShadow: "0 0 12px rgba(189,73,49,0.55)" },
        },
      },
      animation: {
        reveal: "reveal 480ms ease-out backwards",
        "rule-grow": "rule-grow 600ms ease-out backwards",
        "rule-pulse": "rule-pulse 1.8s ease-in-out infinite",
        "math-glow": "math-glow 1.8s ease-in-out infinite",
      },
    },
  },
  plugins: [require("tailwindcss-animate")],
};
