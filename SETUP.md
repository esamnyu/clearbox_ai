# Development Environment Setup Guide

Complete setup instructions for local development and pair programming workflows.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Platform-Specific Setup](#platform-specific-setup)
- [Editor Setup](#editor-setup)
- [Verification](#verification)
- [Pair Programming Setup](#pair-programming-setup)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Required Software

#### Node.js (v16+ required, v18+ recommended)

**macOS (via Homebrew):**
```bash
brew install node@18
```

**macOS (via nvm - recommended for version management):**
```bash
# Install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash

# Restart terminal, then:
nvm install 18
nvm use 18
nvm alias default 18
```

**Linux (Ubuntu/Debian):**
```bash
# Via NodeSource
curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
sudo apt-get install -y nodejs

# Or via nvm (recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 18
```

**Windows:**
```powershell
# Via Chocolatey
choco install nodejs-lts

# Or download installer from https://nodejs.org
```

**Verify Installation:**
```bash
node --version  # Should show v18.x.x or higher
npm --version   # Should show v9.x.x or higher
```

#### Git

**macOS:**
```bash
# Usually pre-installed, or:
brew install git
```

**Linux:**
```bash
sudo apt-get install git
```

**Windows:**
```bash
# Download from https://git-scm.com/download/win
# Or via Chocolatey:
choco install git
```

#### Modern Browser

**Chrome 113+ or Edge 113+ (Recommended):**
- Best WebGPU support
- Best DevTools for Web Workers
- Download: https://www.google.com/chrome/

**Safari 16.4+ (macOS only):**
- Good SharedArrayBuffer support
- Limited WebGPU support

**Firefox:**
- Currently NOT recommended (SharedArrayBuffer restrictions)

### System Requirements

**Minimum:**
- **CPU**: Dual-core 2.0 GHz
- **RAM**: 8 GB
- **Disk**: 5 GB free (2 GB for models + 3 GB for node_modules)
- **OS**: macOS 11+, Windows 10+, Ubuntu 20.04+

**Recommended:**
- **CPU**: Quad-core 3.0 GHz
- **RAM**: 16 GB
- **Disk**: 10 GB free (SSD preferred)
- **GPU**: Not required (CPU inference only)

## Platform-Specific Setup

### macOS Setup

```bash
# 1. Install Xcode Command Line Tools (if not already installed)
xcode-select --install

# 2. Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# 3. Install Node.js
brew install node@18

# 4. Clone repository
git clone <your-repo-url> clearbox_ai
cd clearbox_ai

# 5. Install dependencies
npm install

# 6. Start development server
npm run dev

# 7. Open http://localhost:3001 in Chrome
```

### Linux Setup (Ubuntu/Debian)

```bash
# 1. Update system
sudo apt-get update
sudo apt-get upgrade

# 2. Install build essentials
sudo apt-get install build-essential

# 3. Install Node.js (via nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
source ~/.bashrc
nvm install 18

# 4. Clone repository
git clone <your-repo-url> clearbox_ai
cd clearbox_ai

# 5. Install dependencies
npm install

# 6. Start development server
npm run dev

# 7. Open http://localhost:3001 in Chrome
```

### Windows Setup

**Option 1: Using PowerShell**

```powershell
# 1. Install Chocolatey (if not already installed)
# Run PowerShell as Administrator
Set-ExecutionPolicy Bypass -Scope Process -Force
[System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072
iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# 2. Install Node.js and Git
choco install nodejs-lts git

# 3. Restart PowerShell (as regular user)

# 4. Clone repository
git clone <your-repo-url> clearbox_ai
cd clearbox_ai

# 5. Install dependencies
npm install

# 6. Start development server
npm run dev

# 7. Open http://localhost:3001 in Chrome
```

**Option 2: Using WSL2 (Recommended for advanced users)**

```bash
# 1. Enable WSL2 (PowerShell as Admin)
wsl --install

# 2. Install Ubuntu from Microsoft Store

# 3. Follow Linux setup instructions above in WSL2 terminal
```

## Editor Setup

### VS Code (Recommended)

**Install VS Code:**
- Download from https://code.visualstudio.com/

**Required Extensions:**

```bash
# Open VS Code in project directory
code .

# Install extensions (in VS Code terminal or via UI)
code --install-extension dbaeumer.vscode-eslint
code --install-extension ms-vscode.vscode-typescript-next
code --install-extension bradlc.vscode-tailwindcss
```

**Recommended Extensions:**

```bash
code --install-extension streetsidesoftware.code-spell-checker
code --install-extension usernamehw.errorlens
code --install-extension eamodio.gitlens
code --install-extension ms-vsliveshare.vsliveshare  # For pair programming
```

**Workspace Settings:**

The project includes `.vscode/settings.json` (created in next step) with:
- TypeScript SDK configuration
- ESLint auto-fix on save
- Tailwind CSS IntelliSense
- Recommended formatter settings

### Alternative Editors

**WebStorm / IntelliJ IDEA:**
- Excellent TypeScript support out of the box
- Built-in ESLint integration
- Requires JetBrains license

**Vim / Neovim:**
- Install CoC (Conquer of Completion) or LSP client
- Configure TypeScript language server
- Install ESLint plugin

**Emacs:**
- Use LSP mode with TypeScript language server
- Install Flycheck for ESLint

## Verification

### Step 1: Verify Node.js Installation

```bash
node --version  # v18.x.x or higher
npm --version   # v9.x.x or higher
```

### Step 2: Verify Dependencies

```bash
cd clearbox_ai
npm install

# Should complete without errors
# Total install size: ~800 MB (node_modules)
```

### Step 3: Verify TypeScript

```bash
npx tsc --version  # Version 5.3.x

# Type-check (should pass with no errors)
npx tsc --noEmit
```

### Step 4: Verify Linter

```bash
npm run lint

# Should pass with no errors (or list specific issues to fix)
```

### Step 5: Verify Development Server

```bash
npm run dev

# Should output:
# VITE v5.x.x  ready in XXX ms
#
# ➜  Local:   http://localhost:3001/
# ➜  Network: use --host to expose
```

### Step 6: Verify Browser Compatibility

**Open http://localhost:3001 in Chrome**

1. Open DevTools (F12 or Cmd+Option+I)
2. Check Console for errors
3. Look for these messages:
   - `SharedArrayBuffer is available` (should be true)
   - No CORS errors
   - No CSP (Content Security Policy) errors

**Test Model Loading:**

1. Click "Load Model" button
2. Wait for download (first time: ~500 MB, subsequent: instant from cache)
3. Progress bar should reach 100%
4. Status should change to "Ready"

**Test Tokenization:**

1. Type "Hello world" in input field
2. Verify tokens appear: `["Hello", " world"]`
3. Verify token IDs: `[15496, 995]`

### Step 7: Verify Tests

```bash
npm test

# Should run test suite and pass all tests
```

## Pair Programming Setup

### Local Pair Programming (Same Machine)

**Screen Sharing Options:**
- macOS: Built-in Screen Sharing
- Windows: Quick Assist
- Linux: RustDesk, AnyDesk

**Git Workflow:**
```bash
# Driver commits frequently
git add .
git commit -m "feat(ui): add generation controls"

# Navigator reviews before push
git diff HEAD~1

# Push to shared branch
git push origin feature/session-2
```

### Remote Pair Programming

#### Option 1: VS Code Live Share (Recommended)

**Setup:**

```bash
# Install Live Share extension
code --install-extension ms-vsliveshare.vsliveshare
```

**Usage:**

1. **Driver**: Click "Live Share" in VS Code status bar
2. Copy invitation link
3. **Navigator**: Click link, opens VS Code
4. Both can edit simultaneously
5. Shared terminal available for `npm run dev`

**Audio:** Use Zoom, Discord, or Google Meet for voice

#### Option 2: tmux + SSH (Advanced)

**Setup (Driver machine):**

```bash
# Install tmux
brew install tmux  # macOS
sudo apt-get install tmux  # Linux

# Start shared session
tmux new-session -s pair-programming

# Share session with navigator (via SSH)
# Navigator runs:
ssh user@driver-machine
tmux attach-session -t pair-programming
```

**Usage:**

```bash
# Driver: Terminal 1
npm run dev

# Navigator: Terminal 2
npm run test:watch

# Both can see and control both terminals
```

#### Option 3: Tuple / Pop (Commercial Tools)

- **Tuple**: https://tuple.app/ (macOS/Linux, paid)
- **Pop**: https://pop.com/ (Cross-platform, paid)
- Low latency, high quality screen sharing
- Built-in drawing tools for collaboration

### Git Configuration for Pair Programming

**Co-Authoring Commits:**

```bash
# Commit with both authors
git commit -m "feat(attention): implement attention head detection

Co-authored-by: Researcher Name <researcher@example.com>
Co-authored-by: Engineer Name <engineer@example.com>"
```

**Mob Programming (3+ people):**

```bash
# Install mob tool
brew install remotemobprogramming/brew/mob  # macOS
# Or download from https://mob.sh/

# Start mob session (Driver)
mob start

# Hand off to next driver
mob next

# Navigator joins
mob start
```

## Troubleshooting

### Issue: `npm install` fails

**Error: `EACCES: permission denied`**

```bash
# Fix npm permissions (macOS/Linux)
mkdir ~/.npm-global
npm config set prefix '~/.npm-global'
echo 'export PATH=~/.npm-global/bin:$PATH' >> ~/.bashrc
source ~/.bashrc

# Retry
npm install
```

**Error: `gyp ERR! stack Error: not found: python`**

```bash
# Install Python (required for some native modules)
# macOS:
brew install python@3

# Ubuntu:
sudo apt-get install python3

# Windows:
choco install python
```

### Issue: Dev server fails to start

**Error: `Port 3001 already in use`**

```bash
# Find and kill process using port 3001
# macOS/Linux:
lsof -ti:3001 | xargs kill -9

# Windows:
netstat -ano | findstr :3001
taskkill /PID <PID> /F

# Or use different port
npm run dev -- --port 3002
```

**Error: `COOP/COEP headers not set`**

- Check `vite.config.ts` has correct headers
- Hard refresh browser (Cmd+Shift+R / Ctrl+Shift+R)
- Clear browser cache

### Issue: Model fails to load

**Error: `Failed to download model`**

```bash
# Check internet connection
ping huggingface.co

# Clear browser cache (Chrome):
# DevTools > Application > Storage > Clear site data

# Try different model
# In browser console:
window.localStorage.clear()
```

**Error: `Out of memory`**

- Close other browser tabs
- Restart browser
- Use GPT-2 (124M) instead of GPT-2 Medium (355M)
- Increase system RAM or swap

### Issue: TypeScript errors in editor

**Error: `Cannot find module 'react'`**

```bash
# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install

# Restart TypeScript server in VS Code
# Cmd+Shift+P > "TypeScript: Restart TS Server"
```

**Error: `Type 'X' is not assignable to type 'Y'`**

- Check `tsconfig.json` has `strict: true`
- Verify all dependencies are installed
- Run `npx tsc --noEmit` to see full error context

### Issue: Worker fails to initialize

**Error: `SharedArrayBuffer is not defined`**

- Check browser compatibility (Chrome 113+)
- Verify COOP/COEP headers in Network tab (DevTools)
- Hard refresh browser

**Error: `Worker script failed to load`**

- Check browser console for import errors
- Verify `vite.config.ts` has `worker.format: 'es'`
- Try clearing browser cache

### Issue: Tests fail

**Error: `Module not found`**

```bash
# Update test configuration
npm install -D @vitest/ui happy-dom

# Run tests with verbose output
npm test -- --reporter=verbose
```

### Issue: Pair programming connection fails

**VS Code Live Share not connecting:**

- Check firewall settings
- Try different network (mobile hotspot)
- Update VS Code and Live Share extension
- Restart VS Code

**tmux session not shared:**

- Verify SSH access: `ssh user@driver-machine`
- Check tmux is running: `tmux ls`
- Use absolute session name: `tmux attach -t pair-programming`

## Next Steps

After successful setup:

1. Read [CONTRIBUTING.md](CONTRIBUTING.md) for development workflows
2. Review [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for system design
3. Check [docs/RESEARCHER_GUIDE.md](docs/RESEARCHER_GUIDE.md) for ML research workflows
4. Start with Session 1 verification (model loading + tokenization)

## Getting Help

- **Documentation**: See `docs/` directory
- **Issues**: Open GitHub issue with error details
- **Questions**: Check existing GitHub issues or discussions

Happy developing!
