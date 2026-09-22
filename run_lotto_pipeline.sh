#!/usr/bin/env bash
# =====================================================================
# run_lotto_pipeline.sh
# =====================================================================
# Automates the full UK Lotto NN pipeline workflow:
#   1. Downloads the latest UK Lotto historical feed (CSV) to uk_lotto.csv.
#   2. Runs the Python pipeline (lotto_nn.pipeline), which:
#        - Loads uk_lotto.csv
#        - Extracts the latest actual draw -> actual.txt
#        - Generates NN predictions -> predictions.txt
#        - Runs full backtests -> backtest.txt
#   3. Builds and prints a PREDICTED | ACTUAL | MATCHES matrix.
# =====================================================================
set -euo pipefail

CSV_FILE="uk_lotto.csv"
ACTUAL_FILE="actual.txt"
PREDICTIONS_FILE="predictions.txt"
BACKTEST_FILE="backtest.txt"
NUM_PREDICTIONS="${NUM_PREDICTIONS:-5}"
EPOCHS="${EPOCHS:-50}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
LOTTO_CSV_URL="${LOTTO_CSV_URL:-https://www.national-lottery.co.uk/results/lotto/draw-history/csv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =====================================================================
# SECTION 0 — ENSURE PYTHON DEPENDENCIES ARE INSTALLED
# =====================================================================
if ! python3 -c "import torch, pandas, numpy" >/dev/null 2>&1; then
    echo "Installing required Python dependencies..."
    python3 -m pip install -r "$SCRIPT_DIR/requirements.txt"
fi

# =====================================================================
# SECTION 1 — DOWNLOAD LATEST HISTORICAL FEED
# =====================================================================
echo "Downloading latest UK Lotto historical feed..."
curl -L -o "$CSV_FILE" "$LOTTO_CSV_URL"

if [ ! -s "$CSV_FILE" ]; then
    echo "ERROR: $CSV_FILE is empty or missing."
    exit 1
fi

# =====================================================================
# SECTION 2 — RUN EXISTING PYTHON PIPELINE
# =====================================================================
echo "Running lotto NN pipeline..."
python3 -m lotto_nn.pipeline \
    --csv "$CSV_FILE" \
    --epochs "$EPOCHS" \
    --num-samples "$NUM_SAMPLES" \
    --num-predictions "$NUM_PREDICTIONS" \
    --actual-out "$ACTUAL_FILE" \
    --predictions-out "$PREDICTIONS_FILE" \
    --backtest-out "$BACKTEST_FILE"

if [ ! -s "$ACTUAL_FILE" ] || [ ! -s "$PREDICTIONS_FILE" ]; then
    echo "ERROR: pipeline did not produce $ACTUAL_FILE / $PREDICTIONS_FILE."
    exit 1
fi

# =====================================================================
# SECTION 3 — GENERATE BACKTEST MATRIX (PREDICTED VS ACTUAL)
# =====================================================================
echo ""
echo "PREDICTED | ACTUAL | MATCHES"
echo "--------------------------------"

actual=$(cat "$ACTUAL_FILE")

while read -r pred; do
    matches=$(python3 - "$pred" "$actual" <<'EOF'
import sys

pred = set(sys.argv[1].split())
actual = set(sys.argv[2].split())
print(len(pred & actual))
EOF
)
    echo "$pred | $actual | $matches"
done < "$PREDICTIONS_FILE"
