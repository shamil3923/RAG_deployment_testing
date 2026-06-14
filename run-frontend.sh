#!/usr/bin/env bash
# Start the Vite dev server on http://localhost:5173
set -e
cd "$(dirname "$0")/frontend"

if [ ! -d "node_modules" ]; then
	echo "Installing frontend dependencies (npm ci)..."
	npm ci
fi

exec npm run dev
