#!/bin/zsh
# Every call needed to re-measure the demo against the support case file.
# 23 API calls, about 2-3 minutes. Nothing writes to a GPU-scenario filename.

set -e
cd "$(dirname "$0")"
CASE=case_support.json

if [ -z "$INCEPTION_API_KEY" ]; then
  echo "INCEPTION_API_KEY is not set in this shell. Export it, then re-run." >&2
  exit 1
fi

echo "### 1/4  diffusion, 5 trials  (5 calls) -- states, and the video source"
python3 mercury3_incident_demo.py --mode diffusion --repeat 5 --effort low --no-live \
  --incident "$CASE" \
  --save-states diffusion_states_support.json \
  --results results_diffusion_support.json

echo "\n### 2/4  standard streaming, 5 trials  (5 calls) -- block count"
python3 mercury3_incident_demo.py --mode standard --repeat 5 --effort low \
  --incident "$CASE" --results results_standard_support.json

echo "\n### 3/4  low vs high effort, 5 trials each  (10 calls)"
python3 mercury3_incident_demo.py --mode compare --repeat 5 \
  --incident "$CASE" --results results_compare_support.json

echo "\n### 4/4  the max_completion_tokens trap  (2 calls)"
python3 mercury3_incident_demo.py --mode compare --repeat 1 --max-tokens 1600 \
  --incident "$CASE" --results results_trap_support.json

echo "\n### 1 more call: raw response structure for the article's probe listing"
python3 mercury3_incident_demo.py --mode probe \
  --incident "$CASE" --results results_probe_support.json

echo "\n=== done. 23 calls. ==="
ls -la *_support*.json
