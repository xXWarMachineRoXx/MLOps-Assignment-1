#!/usr/bin/env bash
set -euo pipefail

# Usage: HOST="mlopsassignment1.centralindia.cloudapp.azure.com" ./test-predict.sh
HOST=${HOST:-"mlopsassignment1.centralindia.cloudapp.azure.com"}
URL="http://${HOST}/predict"

# Sample payload; edit as needed
read -r -d '' DATA <<'JSON'
{
  "age": 63,
  "sex": 1,
  "cp": 3,
  "trestbps": 145,
  "chol": 233,
  "fbs": 1,
  "restecg": 0,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 2.3,
  "slope": 0,
  "ca": 0,
  "thal": 1
}
JSON

echo "[info] POST ${URL}"
response=$(curl -sS -X POST "${URL}" \
  -H "Content-Type: application/json" \
  -d "${DATA}")

echo "[info] response:" 
echo "${response}" | jq . || echo "${response}"
