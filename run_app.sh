#!/bin/bash
# Startup script for CrackCapsule Flask app with optimized multiprocessing settings

# Set all threading environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Run the Flask app
echo "🚀 Starting CrackCapsule Flask App with optimized settings..."
python app.py
