#!/usr/bin/env bash
# =====================================================================
# run_lotto_grammar_pipeline.sh
# =====================================================================
# Automates the full UK Lotto grammar-discovery pipeline workflow:
#   1. Downloads the latest UK Lotto historical feed (CSV) to uk_lotto.csv.
#   2. Runs the Python pipeline (lotto_nn.grammar_pipeline), which:
#        - Loads uk_lotto.csv
#        - Extracts a statistical n-gram "grammar" over historical draws
#        - Extracts the latest actual draw -> actual.txt
#        - Generates grammar-aware predictions -> predictions.txt
#        - Runs full backtests -> backtest.txt
#   3. Builds and prints a PREDICTED | ACTUAL | MATCHES matrix.
#
# Note: UK Lotto draws are random and independent; the "grammar" extracted
# here is a descriptive statistical summary only and cannot predict future
# outcomes.
# =====================================================================
set -euo pipefail

CSV_FILE="uk_lotto.csv"
ACTUAL_FILE="actual.txt"
PREDICTIONS_FILE="predictions.txt"
BACKTEST_FILE="backtest.txt"
NUM_PREDICTIONS="${NUM_PREDICTIONS:-5}"
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
LOTTO_CSV_URL="${LOTTO_CSV_URL:-http://lottery.merseyworld.com/cgi-bin/lottery?days=2&Machine=Z&Ballset=0&order=0&show=1&year=0&display=CSV}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# =====================================================================
# SECTION 0 — ENSURE PYTHON DEPENDENCIES ARE INSTALLED
# =====================================================================
if ! python3 -c "import pandas, numpy" >/dev/null 2>&1; then
    echo "Installing required Python dependencies..."
    if ! python3 -m pip --version >/dev/null 2>&1; then
        echo "pip module not found; bootstrapping with ensurepip..."
        python3 -m ensurepip --upgrade
    fi
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
# SECTION 2 — RUN GRAMMAR-DISCOVERY PYTHON PIPELINE
# =====================================================================
echo "Running lotto grammar-discovery pipeline..."
python3 -m lotto_nn.grammar_pipeline \
    --csv "$CSV_FILE" \
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
