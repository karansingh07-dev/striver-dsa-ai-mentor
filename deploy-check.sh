#!/usr/bin/env bash
# Pre-deployment verification script
# Run this before deploying to ensure everything is ready

set -e

echo "=== Striver DSA AI Mentor - Pre-Deployment Check ==="
echo ""

# Check 1: Git status
echo "[1] Git status check..."
if [ -f .env ]; then
    echo "WARNING: .env file exists. Make sure it is in .gitignore."
else
    echo "OK: No .env file in working directory."
fi

if [ -d node_modules ]; then
    echo "WARNING: node_modules/ exists. Should be in .gitignore."
else
    echo "OK: No node_modules in working directory."
fi

if [ -d __pycache__ ]; then
    echo "WARNING: __pycache__/ exists. Should be in .gitignore."
else
    echo "OK: No __pycache__ in working directory."
fi

echo ""

# Check 2: Required files
echo "[2] Required files check..."
required_files=(
    "requirements.txt"
    "server/package.json"
    "client/package.json"
    "Dockerfile.python"
    "Dockerfile.node"
    "Dockerfile.client"
    "docker-compose.yml"
    ".env.example"
    "server/.env.example"
    "DEPLOYMENT.md"
)

for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        echo "OK: $file"
    else
        echo "MISSING: $file"
    fi
done

echo ""

# Check 3: Environment variables
echo "[3] Environment variables check..."
if [ -f .env ]; then
    echo "WARNING: .env exists. Ensure it is not committed to Git."
    grep -E "^(GROQ_API_KEY|GEMINI_API_KEY|OPENAI_API_KEY|QDRANT_API_KEY)=" .env > /dev/null && echo "OK: API keys found in .env" || echo "WARNING: Missing API keys in .env"
else
    echo "INFO: .env not found. Copy .env.example to .env and configure."
fi

echo ""

# Check 4: Python tests
echo "[4] Running Python tests..."
if command -v python &> /dev/null; then
    python -m pytest tests/test_hybrid_search.py -v --tb=short 2>&1 | tail -5
else
    echo "SKIP: Python not available"
fi

echo ""

# Check 5: React build
echo "[5] Building React frontend..."
cd client
if command -v npm &> /dev/null; then
    npm run build 2>&1 | tail -10
else
    echo "SKIP: npm not available"
fi
cd ..

echo ""
echo "=== Pre-deployment check complete ==="
