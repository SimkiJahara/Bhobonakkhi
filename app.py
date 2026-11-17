"""Run: uvicorn app:app --port 8000
Report backend (all free):
  template  - default, no setup, fixed wording based on the scores
  gemini    - set GEMINI_API_KEY (free key from Google AI Studio); writes the report by looking at the photos
  qwen      - REPORT_BACKEND=qwen; runs Qwen2.5-VL-3B locally (needs a GPU; pip install qwen-vl-utils accelerate)
If GEMINI_API_KEY is set, gemini is used automatically. Any failure falls back to the template."""
import io, os, json, joblib, numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from PIL import Image
from features import building_feat, VIEWS

def _load_env(name='.env'):
    """Read KEY=value lines from .env next to this file (no extra package needed)."""
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
    if os.path.exists(p):
        for line in open(p):
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                k, v = line.split('=', 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"\''))
_load_env()

M = joblib.load('bhobonakkhi_model.joblib'); INDS = M['inds']
app = FastAPI()

ISSUE = {
 'Soft_Story': 'Open or weak ground floor (shopfront, parking, stilts) that can fail first in an earthquake.',
 'Cracks': 'Visible cracking in walls or wall-column junctions that suggests structural distress.',
 'Plan_Irregularity': 'Irregular footprint (L/U-shape, setbacks) that causes twisting and stress concentration.',
 'Material_Degradation': 'Surface and material deterioration (plaster loss, staining, rust) that weakens the structure.',
 'Pounding': 'Little or no gap to neighbouring buildings, so they may hit each other during shaking.',
 'Overhang': 'Projecting balconies or upper floors that add load and are vulnerable to failure.'}

def to_class(t): return 'Low' if t < 5 else 'Moderate' if t < 10 else 'High' if t < 14 else 'Critical'

def template_report(r):
    hi = [i for i in INDS if r['indicators'][i] >= 1.5]
    lines = [f"Overall risk: {r['class']} (score {r['total']:.1f} / 18).", '']
    if hi:
        lines += ['Main problems indicated:'] + [f"- {i.replace('_', ' ')} ({r['indicators'][i]:.1f}/3): {ISSUE[i]}" for i in hi]
    else:
        lines += ['No single indicator scores high; the building looks fairly regular and sound from the images.']
    lines += ['', 'This is an automated screening result. A qualified engineer should inspect the building before any decision.']
    return '\n'.join(lines)

RUBRIC = {
 'Soft_Story': '0 solid infill walls at ground floor like upper floors; 1 ground floor mostly enclosed, one open bay; 2 open shopfront/parking with columns showing; 3 ground floor fully open (stilts/columns only)',
 'Cracks': '0 none visible; 1 a few isolated hairline cracks; 2 cracks at several places, moderate width; 3 wide/diagonal structural cracks',
 'Plan_Irregularity': '0 regular rectangular footprint; 1 slight asymmetry/small setback; 2 clear L/U shape or re-entrant corner; 3 complex multi-wing plan',
 'Material_Degradation': '0 finish intact; 1 surface weathering/discoloration; 2 plaster loss, exposed brick, rust; 3 major spalling, exposed corroded rebar',
 'Pounding': '0 clear separation (>1 m), no tall neighbour; 1 adequate gap but taller/stiffer neighbour nearby; 2 narrow gap to neighbour; 3 touching/no gap',
 'Overhang': '0 none; 1 small balconies/sunshades; 2 large cantilevered balconies or projecting upper floors; 3 heavy cantilevered mass'}

def blind_prompt():
    rub = '\n'.join(f'- {k}: {v}' for k, v in RUBRIC.items())
    return ('You are an assistant for visual seismic screening of buildings in Dhaka. The images above are 4 drone views '
            '(facade, oblique45, oblique60, nadir) of ONE building. Judge each indicator only from what is visible.\n'
            f'Scoring rubric (0-3):\n{rub}\n'
            'Reply with JSON only: {"indicators": {"<indicator>": {"score": 0-3 or null, "evidence": "one short sentence on what you see", '
            '"confidence": "low|medium|high"}, ...all six...}, "summary": "2-3 sentence plain summary", '
            '"next_steps": ["2-4 short inspection recommendations"]}. Use null if the images give no evidence. Do not invent hidden conditions.')

def small(im, m=1024):
    im = im.convert('RGB'); im.thumbnail((m, m)); return im

def gemini_raw(images):
    from google import genai
    from google.genai import types
    parts = []
    for v in VIEWS: parts += [f'View: {v}', small(images[v])]
    parts.append(blind_prompt())
    c = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
    return c.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-3.8-flash'), contents=parts,
            config=types.GenerateContentConfig(response_mime_type='application/json')).text

_q = None
def qwen_raw(images):
    global _q
    from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
    from qwen_vl_utils import process_vision_info
    if _q is None:
        mid = 'Qwen/Qwen2.5-VL-3B-Instruct'
        _q = (Qwen2_5_VLForConditionalGeneration.from_pretrained(mid, torch_dtype='auto', device_map='auto'),
              AutoProcessor.from_pretrained(mid, min_pixels=256*28*28, max_pixels=640*28*28))
    model, proc = _q
    content = []
    for v in VIEWS: content += [{'type': 'text', 'text': f'View: {v}'}, {'type': 'image', 'image': small(images[v])}]
    content.append({'type': 'text', 'text': blind_prompt()})
    msgs = [{'role': 'user', 'content': content}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    im, vid = process_vision_info(msgs)
    inp = proc(text=[text], images=im, videos=vid, padding=True, return_tensors='pt').to(model.device)
    out = model.generate(**inp, max_new_tokens=900)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]

def parse_json(t):
    t = t.strip()
    i, j = t.find('{'), t.rfind('}')
    return json.loads(t[i:j + 1])

def vlm_assess(images):
    be = os.getenv('REPORT_BACKEND') or ('gemini' if os.getenv('GEMINI_API_KEY') else 'template')
    try:
        if be == 'gemini': return parse_json(gemini_raw(images)), 'gemini'
        if be == 'qwen': return parse_json(qwen_raw(images)), 'qwen'
    except Exception as e:
        print('report backend failed, using template:', e)
    return None, 'template'

def finish(r, vlm, source):
    """Compare model vs independent visual check, build warnings and the written report."""
    warns = []
    low_but = [i for i in INDS if r['indicators'][i] >= 1.5]
    if low_but and r['class'] in ('Low', 'Moderate'):
        warns.append(f"Overall class is {r['class']}, but elevated indicator(s): " +
                     ', '.join(f"{i.replace('_', ' ')} ({r['indicators'][i]:.1f}/3)" for i in low_but) + '.')
    if not vlm:
        r['comparison'] = []; r['report'] = template_report(r)
    else:
        cmp_, lines = [], [f"Overall risk: {r['class']} (model score {r['total']:.1f} / 18).", '', vlm.get('summary', ''), '']
        for i in INDS:
            v = (vlm.get('indicators') or {}).get(i) or {}
            vs = v.get('score'); ms = r['indicators'][i]
            diff = None if vs is None else round(abs(float(vs) - ms), 2)
            status = 'no evidence' if diff is None else ('agree' if diff <= 1 else 'disagree')
            if status == 'disagree':
                warns.append(f"Model ({ms:.1f}) and visual check ({vs}) disagree on {i.replace('_', ' ')}. Verify this one manually.")
            cmp_.append({'indicator': i, 'model': ms, 'vlm': vs, 'status': status, 'evidence': v.get('evidence', ''),
                         'confidence': v.get('confidence', '')})
        steps = vlm.get('next_steps') or []
        lines += ['Recommended next steps:'] + [f'{n}. {t}' for n, t in enumerate(steps, 1)] + \
                 ['', 'Screening aid only; not an engineering assessment.']
        r['comparison'] = cmp_; r['report'] = '\n'.join(lines)
    r['warnings'] = warns; r['report_source'] = source
    return r

def predict(x):
    z = M['scaler'].transform(x)
    ind = {i: float(np.clip(M['reg_ind'][i].predict(z)[0], 0, 3)) for i in INDS}
    total = sum(ind.values()); p_low = float(M['clf'].predict_proba(z)[0, 1])
    cls = 'Low' if p_low >= 0.5 else to_class(max(total, 5))       # same rule as your notebook's hybrid_class
    return {'indicators': {k: round(v, 2) for k, v in ind.items()}, 'total': round(total, 2),
            'class': cls, 'p_low': round(p_low, 2)}

@app.post('/analyze')
async def analyze(facade: UploadFile = File(...), oblique45: UploadFile = File(...),
                  oblique60: UploadFile = File(...), nadir: UploadFile = File(...)):
    try:
        images = {v: Image.open(io.BytesIO(await f.read())) for v, f in
                  zip(VIEWS, [facade, oblique45, oblique60, nadir])}
    except Exception:
        raise HTTPException(400, 'Could not read one of the images.')
    r = predict(building_feat(images)[None])
    vlm, src = vlm_assess(images)
    return finish(r, vlm, src)

@app.get('/')
def index(): return FileResponse('index.html')