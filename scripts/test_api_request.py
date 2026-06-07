from __future__ import annotations

import json
from pathlib import Path
import requests

api_url = "http://127.0.0.1:8000"
payload = json.loads((Path(__file__).resolve().parents[1] / "sample_data" / "sample_patient.json").read_text())

print("Health:")
print(requests.get(f"{api_url}/health", timeout=5).json())

print("Prediction:")
r = requests.post(f"{api_url}/predict", json=payload, timeout=30)
print(r.status_code)
print(json.dumps(r.json(), indent=2))
