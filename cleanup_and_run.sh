#!/bin/bash
# cleanup_and_run.sh

# ./cleanup_and_run.sh run_comparison.py --n_sims ##

echo "Cleaning Python cache..."
find . -name "*.pyc" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null

echo "Clearing system caches..."
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # Linux
    sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' 2>/dev/null || echo "Could not clear system cache (need sudo)"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sudo purge 2>/dev/null || echo "Could not clear system cache (need sudo)"
fi

echo "Starting fresh Python process..."
python -B "$@"