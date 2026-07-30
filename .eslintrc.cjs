/**
 * ESLint config for NeuroScope-Web.
 *
 * `npm run lint` was in package.json since the first commit but no config file
 * ever existed, so the script has always exited non-zero with "couldn't find a
 * configuration file" — i.e. lint has never actually run on this repo.
 *
 * Flat config (eslint.config.js) is the modern format, but the pinned ESLint
 * here is 8.57, where flat config is still opt-in behind an env var. This uses
 * the eslintrc format that 8.x reads by default. `.cjs` because package.json
 * sets "type": "module".
 *
 * Type-aware rules (parserOptions.project) are deliberately NOT enabled: they
 * roughly triple lint time and duplicate what `tsc --noEmit` already gates in
 * `npm run build`. Syntactic rules here, semantic ones in tsc.
 */
module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    worker: true,
  },
  extends: ["eslint:recommended", "plugin:@typescript-eslint/recommended"],
  parser: "@typescript-eslint/parser",
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
    ecmaFeatures: { jsx: true },
  },
  plugins: ["@typescript-eslint", "react-hooks"],
  settings: {
    react: { version: "18.2" },
  },
  ignorePatterns: [
    "dist",
    "node_modules",
    "coverage",
    // Stale agent worktrees; vite.config.ts excludes them from Vitest too.
    ".claude",
    "*.cjs",
  ],
  rules: {
    ...require("eslint-plugin-react-hooks").configs.recommended.rules,

    // Unused locals are a build error via tsc's noUnusedLocals; keep the lint
    // signal but let an underscore prefix mark a deliberate discard.
    "@typescript-eslint/no-unused-vars": [
      "error",
      { argsIgnorePattern: "^_", varsIgnorePattern: "^_" },
    ],

    // `any` is a warning, not an error: worker.ts needs a couple of genuine
    // escape hatches at the transformers.js boundary (documented inline), and
    // failing the build on them would push people toward silent @ts-ignore.
    "@typescript-eslint/no-explicit-any": "warn",

    // The analysis layer is numeric code where `console` is debugging noise,
    // but the worker legitimately logs load progress. Warn, don't fail.
    "no-console": ["warn", { allow: ["warn", "error"] }],

    // Enforced by CLAUDE.md's coding standards.
    "prefer-const": "error",
    "no-var": "error",
    radix: "error",
    eqeqeq: ["error", "always", { null: "ignore" }],
  },
  overrides: [
    {
      // Vitest globals + looser rules for fixtures and specs.
      files: [
        "tests/**/*.ts",
        "tests/**/*.tsx",
        "**/*.test.ts",
        "**/*.test.tsx",
      ],
      env: { node: true },
      globals: {
        describe: "readonly",
        it: "readonly",
        expect: "readonly",
        beforeEach: "readonly",
        afterEach: "readonly",
        vi: "readonly",
      },
      rules: {
        "@typescript-eslint/no-explicit-any": "off",
        "no-console": "off",
      },
    },
  ],
};
