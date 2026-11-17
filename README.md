# BhobonAkkhi

AI-assisted seismic vulnerability screening for buildings in Dhaka. Upload four drone views of a building
(façade, oblique 45°, oblique 60°, nadir) to get a risk rating (Low / Moderate / High / Critical), six indicator scores,
and a written report.

**Pipeline:** DINOv2 features → ridge regression per indicator (+ Low-vs-rest classifier) → independent visual check by a
vision-language model (Gemini free tier, or local Qwen2.5-VL) → comparison and report.

> Screening aid only. Not a structural engineering assessment. Trained on 55 buildings.

## Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then put your Gemini key in .env (never commit it)
uvicorn app:app --port 8000
```
Open http://localhost:8000. Without a key the app falls back to a simple template report.