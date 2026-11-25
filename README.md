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

## Features
- **Live progress**: each step (reading photos, model, AI check, report) ticks off while it runs.
- **Model vs AI radar chart** and a per-indicator comparison. `FINAL_SCORE=average` (default) uses the mean of the
  model and the AI check; `FINAL_SCORE=model` uses the model only.
- **Evidence on the photos**: Gemini marks where it saw each problem; click an indicator to highlight it.
- **Heatmap** (on demand, ~1-2 min on CPU): occlusion map showing which parts of each photo pushed the model's score up or down.
- **History and compare**: every analysis is saved in `data/` (SQLite + photos); tick two to compare side by side.
- **PDF report** with annotated photos, radar chart, indicator table and written report.
- **Email**: send the PDF to up to 5 addresses (needs `SMTP_USER` and `SMTP_PASSWORD` in `.env`).

`data/` holds saved analyses and photos. Keep it out of git (add `data/` and `.env` to `.gitignore`).

## Evaluation (capstone)
```bash
python evaluate_vlm.py /path/to/All /path/to/labels.csv
```
Runs the blind Gemini check on every labelled building (cached and resumable), then compares the VLM alone, the
cross-validated ridge model, and their average against your labels. CSV tables are written to `results/`.