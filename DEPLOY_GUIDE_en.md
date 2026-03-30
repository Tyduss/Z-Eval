# Z-Eval Deployment Guide

> This document supplements the official README with solutions to common deployment issues.

## 1. Environment Requirements

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | **3.10 - 3.12** | ⚠️ **Python 3.14+ is NOT supported** due to pydantic v1 and langchain incompatibility |
| Node.js | 18+ | Frontend dependency |
| npm/pnpm | Latest | Package manager |

## 2. Backend Deployment

### 2.1 Create Python Environment (Conda Recommended)

```bash
# If using conda for the first time, accept the terms of service first
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create Python 3.12 environment
conda create -n one-eval python=3.12 -y
conda activate one-eval
```

### 2.2 Install Project Dependencies

```bash
cd /path/to/One-Eval
pip install -e .
```

### 2.3 Verify Installation

```bash
# Confirm Python path is correct (should be conda env path)
which python3
# Expected: /opt/homebrew/Caskroom/miniconda/base/envs/one-eval/bin/python3
# ❌ Wrong: /Library/Frameworks/Python.framework/Versions/3.14/bin/python3

# Verify open-dataflow is correctly installed
python -c "from dataflow.operators.core_text import BenchAnswerGenerator; print('OK')"
# Expected output: OK (may show some INFO logs)

# If ImportError, upgrade open-dataflow
pip install --upgrade open-dataflow
```

### 2.4 Start Backend Service

```bash
# ⚠️ Important: use python -m instead of calling uvicorn directly
# This ensures the conda environment's Python is used
python -m uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000

# ❌ Wrong way (may call system Python's uvicorn)
# uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000
```

### 2.5 Verify Backend is Running

```bash
# Visit API docs
open http://127.0.0.1:8000/docs
```

## 3. Frontend Deployment

```bash
cd one-eval-web
npm install
npm run dev
```

Visit http://localhost:5173

## 4. Troubleshooting

### Q1: `ImportError: cannot import name 'BenchAnswerGenerator'`

**Cause**: `open-dataflow` version too low or installed in wrong Python environment

**Fix**:
```bash
# Check install location
pip show open-dataflow

# Location should be conda env path, e.g.:
# /opt/homebrew/Caskroom/miniconda/base/envs/one-eval/lib/python3.12/site-packages

# If it's a different path (e.g. /Library/Frameworks/...), it was installed in the wrong env
# Reinstall:
conda activate one-eval
pip install -e . --force-reinstall
```

### Q2: `Core Pydantic V1 functionality isn't compatible with Python 3.14`

**Cause**: Python version too high (3.14+)

**Fix**: Use Python 3.12 or lower when creating the conda environment

### Q3: Module not found when starting backend

**Cause**: Dependencies installed in system Python instead of conda environment

**Fix**:
```bash
# Confirm current Python
which python3

# If not conda env path, reactivate
conda activate one-eval

# Start using python -m
python -m uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000
```

### Q4: TOS error when creating conda environment

**Fix**:
```bash
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
```

## 5. Full Startup Flow

```bash
# 1. Activate environment
conda activate one-eval

# 2. Start backend (Terminal 1)
cd /path/to/One-Eval
python -m uvicorn one_eval.server.app:app --host 127.0.0.1 --port 8000

# 3. Start frontend (Terminal 2)
cd /path/to/One-Eval/one-eval-web
npm run dev

# 4. Visit http://localhost:5173
# 5. On first use, go to Settings to configure API Key, model, and other parameters
```

## 6. Features

- **Task State Persistence**: Switching tabs won't lose task progress; returning to the workspace automatically resumes polling
