"""Run: uvicorn app:app --port 8000
Report backend (all free):
  template  - default, no setup, fixed wording based on the scores
  gemini    - set GEMINI_API_KEY (free key from Google AI Studio); looks at the photos, scores them and marks evidence
  qwen      - REPORT_BACKEND=qwen; runs Qwen2.5-VL-3B locally (needs a GPU; pip install qwen-vl-utils accelerate)
If GEMINI_API_KEY is set, gemini is used automatically. Any failure falls back to the template.

Final score (.env FINAL_SCORE):
  average   - default; each indicator = mean of the model and the AI visual check (model alone where the AI saw nothing)
  model     - the ridge model only; the AI check is shown for comparison but does not change the scores
The Low class always comes from the Low-vs-rest classifier, as in the notebook's hybrid_class.

Email (.env): SMTP_USER, SMTP_PASSWORD (Gmail app password). Optional SMTP_HOST, SMTP_PORT.
  AUTO_EMAIL_TO   - every analysis that is NOT Low risk is emailed here automatically
                    (default umme.simki@northsouth.edu; several addresses separated by commas; "off" to disable).
                    Low-risk reports are only sent when someone presses "Email report".
  AUTO_EMAIL_LANG - language of the PDF in the automatic email: en (default), bn, or both.

Bangla: the report, warnings and evidence are also written in Bangla (report_bn, warnings_bn, evidence_bn).
The fixed parts come from Bangla templates; the AI's own sentences are translated with one extra text-only
Gemini call when GEMINI_API_KEY is set (otherwise the Bangla report falls back to the template)."""
import io, os, re, json, time, traceback, joblib, numpy as np
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Body
from fastapi.responses import FileResponse, StreamingResponse, Response
from PIL import Image
from features import image_feat, VIEWS
import storage

BASE = os.path.dirname(os.path.abspath(__file__))

def _load_env(name='.env'):
    """Read KEY=value lines from .env next to this file (no extra package needed)."""
    p = os.path.join(BASE, name)
    if os.path.exists(p):
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    k, v = line.split('=', 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"\''))
_load_env()

M = joblib.load(os.path.join(BASE, 'bhobonakkhi_model.joblib')); INDS = M['inds']
SCORE_MODE = 'model' if os.getenv('FINAL_SCORE', 'average').strip().lower() == 'model' else 'average'
app = FastAPI()

ISSUE = {
 'Soft_Story': 'Open or weak ground floor (shopfront, parking, stilts) that can fail first in an earthquake.',
 'Cracks': 'Visible cracking in walls or wall-column junctions that suggests structural distress.',
 'Plan_Irregularity': 'Irregular footprint (L/U-shape, setbacks) that causes twisting and stress concentration.',
 'Material_Degradation': 'Surface and material deterioration (plaster loss, staining, rust) that weakens the structure.',
 'Pounding': 'Little or no gap to neighbouring buildings, so they may hit each other during shaking.',
 'Overhang': 'Projecting balconies or upper floors that add load and are vulnerable to failure.'}

# ---------------------------------------------------------------- Bangla text
BN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
def bd(x): return str(x).translate(BN_DIGITS)
BN_CLS = {'Low': 'নিম্ন', 'Moderate': 'মাঝারি', 'High': 'উচ্চ', 'Critical': 'অতি উচ্চ'}
BN_IND = {'Soft_Story': 'সফট স্টোরি', 'Cracks': 'ফাটল', 'Plan_Irregularity': 'নকশার অনিয়ম',
          'Material_Degradation': 'উপকরণের ক্ষয়', 'Pounding': 'পাউন্ডিং', 'Overhang': 'ওভারহ্যাং'}
ISSUE_BN = {
 'Soft_Story': 'নিচতলা খোলা বা দুর্বল (দোকান, পার্কিং বা শুধু কলামের উপর দাঁড়ানো), যা ভূমিকম্পে সবার আগে ভেঙে পড়তে পারে।',
 'Cracks': 'দেয়ালে বা দেয়াল-কলামের সংযোগে দৃশ্যমান ফাটল, যা কাঠামোগত সমস্যার ইঙ্গিত দেয়।',
 'Plan_Irregularity': 'অনিয়মিত নকশা (L বা U আকৃতি, সেটব্যাক), যা কম্পনের সময় মোচড় তৈরি করে এবং কিছু জায়গায় চাপ বাড়ায়।',
 'Material_Degradation': 'উপকরণের ক্ষয় (প্লাস্টার খসে পড়া, দাগ, মরিচা), যা কাঠামোকে দুর্বল করে।',
 'Pounding': 'পাশের ভবনের সাথে ফাঁক খুব কম বা নেই, তাই কম্পনের সময় ভবনগুলো একে অপরকে ধাক্কা দিতে পারে।',
 'Overhang': 'বাইরে বেরিয়ে থাকা বারান্দা বা উপরের তলা, যা বাড়তি ভার যোগ করে এবং ভেঙে পড়ার ঝুঁকিতে থাকে।'}
BN_MODE = {'average': 'মডেল ও এআই চাক্ষুষ যাচাইয়ের গড়', 'model': 'মডেল থেকে'}
BN_DISCLAIMER = 'এটি একটি স্বয়ংক্রিয় স্ক্রিনিং ফলাফল। কোনো সিদ্ধান্ত নেওয়ার আগে একজন যোগ্য প্রকৌশলীকে দিয়ে ভবনটি পরিদর্শন করানো উচিত।'

def template_report_bn(r):
    hi = [i for i in INDS if r['indicators'][i] >= 1.5]
    lines = [f"সামগ্রিক ঝুঁকি: {BN_CLS[r['class']]} (স্কোর {bd(format(r['total'], '.1f'))} / ১৮)।", '']
    if hi:
        lines += ['প্রধান যে সমস্যাগুলো দেখা গেছে:'] + \
                 [f"- {BN_IND[i]} ({bd(format(r['indicators'][i], '.1f'))}/৩): {ISSUE_BN[i]}" for i in hi]
    else:
        lines += ['কোনো একক সূচকের স্কোর বেশি নয়; ছবিতে ভবনটি মোটামুটি নিয়মিত ও ভালো অবস্থায় দেখা যাচ্ছে।']
    lines += ['', BN_DISCLAIMER]
    return '\n'.join(lines)

def warnings_bn(r):
    final, out = r['indicators'], []
    hi = [i for i in INDS if final[i] >= 1.5]
    if hi and r['class'] in ('Low', 'Moderate'):
        out.append(f"সামগ্রিক শ্রেণি {BN_CLS[r['class']]} ঝুঁকি, কিন্তু কিছু সূচকের স্কোর বেশি: " +
                   ', '.join(f"{BN_IND[i]} ({bd(format(final[i], '.1f'))}/৩)" for i in hi) + '।')
    for c in r.get('comparison') or []:
        if c['status'] == 'disagree':
            out.append(f"{BN_IND[c['indicator']]} নিয়ে মডেল ({bd(format(c['model'], '.1f'))}) ও এআই চাক্ষুষ যাচাই "
                       f"({bd(format(c['vlm'], 'g'))}) একমত নয়। এটি নিজে যাচাই করুন।")
    return out

def add_bangla(r, tr=None):
    """Fill report_bn, warnings_bn and evidence_bn. tr = Gemini translation of the AI's sentences (or None)."""
    r['warnings_bn'] = warnings_bn(r)
    if tr:
        for c in r.get('comparison') or []:
            t = tr['evidence'].get(c['indicator'])
            if t and c.get('evidence'): c['evidence_bn'] = t
    if r.get('report_source') in ('gemini', 'qwen') and tr and tr.get('summary'):
        mode = r.get('score_mode', 'model')
        lines = [f"সামগ্রিক ঝুঁকি: {BN_CLS[r['class']]} (স্কোর {bd(format(r['total'], '.1f'))} / ১৮, {BN_MODE[mode]})।",
                 '', tr['summary'], '']
        if tr.get('next_steps'):
            lines += ['পরবর্তী করণীয়:'] + [f'{n}. {t}' for n, t in enumerate(tr['next_steps'], 1)] + ['']
        lines += ['শুধু স্ক্রিনিং সহায়ক; প্রকৌশল মূল্যায়ন নয়।']
        r['report_bn'] = '\n'.join(lines)
    else:
        rep = template_report_bn(r)
        if r.get('report_source') in ('gemini', 'qwen'):
            rep += '\n\n(এআই-এর বিস্তারিত সারাংশ বাংলায় পাওয়া যায়নি; সেটি ইংরেজি প্রতিবেদনে আছে।)'
        r['report_bn'] = rep
    return r

def to_class(t): return 'Low' if t < 5 else 'Moderate' if t < 10 else 'High' if t < 14 else 'Critical'

def mode_text(mode):
    return 'average of the model and the AI visual check' if mode == 'average' else 'from the model'

def template_report(r):
    hi = [i for i in INDS if r['indicators'][i] >= 1.5]
    lines = [f"Overall risk: {r['class']} (score {r['total']:.1f} / 18).", '']
    if hi:
        lines += ['Main problems indicated:'] + [f"- {i.replace('_', ' ')} ({r['indicators'][i]:.1f}/3): {ISSUE[i]}" for i in hi]
    else:
        lines += ['No single indicator scores high; the building looks fairly regular and sound from the images.']
    lines += ['', 'This is an automated screening result. A qualified engineer should inspect the building before any decision.']
    return '\n'.join(lines)

# Exactly the project's scoring sheet (structural_indicator.pdf): which views to check, and what 0-3 mean.
RUBRIC = {
 'Soft_Story': dict(views=['facade', 'oblique45', 'oblique60'], levels=[
    'Solid infill walls at ground floor, same as upper floors',
    'Ground floor mostly enclosed, one open bay (e.g. a garage door)',
    'Open shopfront/parking with columns showing, partial infill elsewhere',
    'Ground floor fully open: stilts/columns only, no infill at all']),
 'Cracks': dict(views=['facade', 'oblique45', 'oblique60'], levels=[
    'No cracks visible at this resolution',
    'A few isolated hairline cracks',
    'Cracks at multiple locations, moderate width, e.g. at wall-column junctions',
    'Wide/diagonal structural cracks, cracking through slabs or load-bearing walls']),
 'Plan_Irregularity': dict(views=['nadir', 'oblique45', 'oblique60'], levels=[
    'Regular rectangular/symmetric footprint',
    'Slight asymmetry, small setback',
    'Clear L/U-shape, re-entrant corner, uneven extension',
    'Complex multi-wing plan, severe discontinuity between floors']),
 'Material_Degradation': dict(views=['facade', 'oblique45', 'oblique60'], levels=[
    'Finish looks intact, no staining',
    'Surface weathering, minor discoloration',
    'Patches of plaster loss/exposed brick, some rust on railings',
    'Major spalling, exposed corroded rebar, rotten structural timber']),
 'Pounding': dict(views=['nadir', 'oblique45', 'oblique60'], levels=[
    'Clear separation (>1 m), no tall neighbour',
    'Adequate gap but a taller/stiffer neighbour nearby',
    'Narrow gap, floor levels between buildings do not align',
    'Zero gap/shared party wall with a structurally different neighbour']),
 'Overhang': dict(views=['facade', 'oblique45', 'oblique60'], levels=[
    'No cantilevers',
    'Small balcony, clearly supported',
    'Cantilevered slab with no visible bracket/support',
    'Large heavy cantilever (extra room) unsupported, especially on upper floors'])}
NOTES = {'Plan_Irregularity': 'judge the footprint shape on nadir, cross-check massing on the obliques',
         'Pounding': 'gap width on nadir, height mismatch on the obliques',
         'Soft_Story': 'the ground floor is rarely visible from nadir'}

def blind_prompt():
    rub = '\n'.join(f"- {k} (check views: {', '.join(v['views'])}{'; ' + NOTES[k] if k in NOTES else ''}): "
                    + '; '.join(f'{n} = {t}' for n, t in enumerate(v['levels'])) for k, v in RUBRIC.items())
    return ('You are an assistant for visual seismic screening of buildings in Dhaka. The images above are 4 drone views '
            '(facade, oblique45, oblique60, nadir) of ONE building. Judge each indicator only from what is visible, '
            'using only the views listed for it.\n'
            f'Scoring rubric (0-3):\n{rub}\n'
            'Reply with JSON only: {"indicators": {"<indicator>": {"score": 0-3 or null, '
            '"evidence": "one short sentence on what you see", "confidence": "low|medium|high", '
            '"view": "the one view, from that indicator\'s check views, where the evidence is clearest, or null", '
            '"box_2d": [ymin, xmin, ymax, xmax] normalised to 0-1000, or null}, ...all six...}, '
            '"summary": "2-3 sentence plain summary", "next_steps": ["2-4 short inspection recommendations"]}.\n'
            'box_2d must tightly enclose the specific thing you scored (the crack, the stained patch, the open ground '
            'floor, the gap between buildings, the cantilever, the footprint). Never box the whole building or the whole '
            'photo unless the indicator is about the whole footprint. If the score is 0 or there is nothing specific to '
            'point at, use null for view and box_2d. Use null for score if the images give no evidence. '
            'Do not invent hidden conditions.')

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
            config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0)).text

def translate_bn(vlm):
    """Translate the AI's own sentences (summary, next steps, evidence) into Bangla with one text-only Gemini call.
    Kept separate from the blind check so the scoring prompt stays exactly as evaluated. Returns None on failure."""
    if not vlm or not os.getenv('GEMINI_API_KEY'):
        return None
    src = {'summary': vlm['summary'], 'next_steps': vlm['next_steps'],
           'evidence': {i: vlm['indicators'][i]['evidence'] for i in INDS if vlm['indicators'][i]['evidence']}}
    prompt = ('Translate every text value in this JSON from English into clear, natural Bangla (Bengali script) for a '
              'building-safety screening report read by ordinary people in Dhaka. Keep the same keys and the same '
              'number of list items. Write numbers with Bengali digits. Technical words that people in Dhaka usually '
              'say in English (soft story, cantilever, column, beam, plaster) may be written in Bengali script. '
              'Reply with JSON only.\n\n' + json.dumps(src, ensure_ascii=False))
    try:
        from google import genai
        from google.genai import types
        c = genai.Client(api_key=os.environ['GEMINI_API_KEY'])
        t = c.models.generate_content(model=os.getenv('GEMINI_MODEL', 'gemini-3.8-flash'), contents=[prompt],
                config=types.GenerateContentConfig(response_mime_type='application/json', temperature=0)).text
        out = parse_json(t)
        ev = out.get('evidence') if isinstance(out.get('evidence'), dict) else {}
        steps = out.get('next_steps') if isinstance(out.get('next_steps'), list) else []
        return {'summary': str(out.get('summary') or ''),
                'next_steps': [str(s) for s in steps if s][:6] if len(steps) == len(src['next_steps']) else [],
                'evidence': {i: str(ev[i]) for i in INDS if isinstance(ev.get(i), str) and ev[i].strip()}}
    except Exception as e:
        print('Bangla translation failed, using the Bangla template:', e)
        return None

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
    out = model.generate(**inp, max_new_tokens=1200)
    return proc.batch_decode(out[:, inp.input_ids.shape[1]:], skip_special_tokens=True)[0]

def parse_json(t):
    t = t.strip()
    i, j = t.find('{'), t.rfind('}')
    return json.loads(t[i:j + 1])

def clean_vlm(vlm, source):
    """Make the AI reply safe to use: scores clamped to 0-3, boxes checked, missing parts filled."""
    if not isinstance(vlm, dict):
        return None
    raw = vlm.get('indicators') if isinstance(vlm.get('indicators'), dict) else {}
    low = {str(k).lower(): v for k, v in raw.items()}
    out = {}
    for i in INDS:
        v = raw.get(i) or low.get(i.lower()) or {}
        v = v if isinstance(v, dict) else {}
        try:
            s = None if v.get('score') is None else min(3.0, max(0.0, float(v['score'])))
        except (TypeError, ValueError):
            s = None
        view = v.get('view') if v.get('view') in RUBRIC[i]['views'] else None   # only the views the sheet allows
        box = None
        if view and source == 'gemini':        # Qwen uses pixel coordinates of its own resized input, so skip its boxes
            try:
                b = [min(1000.0, max(0.0, float(x))) for x in v.get('box_2d')]
                area = (b[2] - b[0]) * (b[3] - b[1]) / 1e6 if len(b) == 4 else 0
                big_ok = i == 'Plan_Irregularity'        # a footprint box is naturally large
                if len(b) == 4 and b[2] - b[0] >= 5 and b[3] - b[1] >= 5 and (area <= 0.3 or big_ok):
                    box = [round(x, 1) for x in b]
            except (TypeError, ValueError):
                pass
        out[i] = dict(score=s, evidence=str(v.get('evidence') or ''), confidence=str(v.get('confidence') or ''),
                      view=view, box=box)
    steps = vlm.get('next_steps') if isinstance(vlm.get('next_steps'), list) else []
    return dict(indicators=out, summary=str(vlm.get('summary') or ''), next_steps=[str(s) for s in steps if s][:6])

def vlm_assess(images):
    be = os.getenv('REPORT_BACKEND') or ('gemini' if os.getenv('GEMINI_API_KEY') else 'template')
    try:
        if be == 'gemini': return clean_vlm(parse_json(gemini_raw(images)), 'gemini'), 'gemini'
        if be == 'qwen': return clean_vlm(parse_json(qwen_raw(images)), 'qwen'), 'qwen'
    except Exception as e:
        print('report backend failed, using template:', e)
    return None, 'template'

def finish(r, vlm, source, tr=None):
    """Combine model and AI check into the final scores, then build warnings and the written report."""
    mode = SCORE_MODE if vlm else 'model'
    model = dict(r['indicators'])
    final, cmp_ = {}, []
    for i in INDS:
        ms = model[i]
        v = vlm['indicators'][i] if vlm else {}
        vs = v.get('score')
        final[i] = round((ms + vs) / 2, 2) if (mode == 'average' and vs is not None) else ms
        if vlm:
            diff = None if vs is None else round(abs(vs - ms), 2)
            status = ('no evidence' if diff is None else 'exact' if round(ms) == round(vs)
                      else 'close' if diff <= 1 else 'disagree')
            cmp_.append({'indicator': i, 'model': ms, 'vlm': vs, 'final': final[i], 'status': status,
                         'evidence': v.get('evidence', ''), 'confidence': v.get('confidence', ''),
                         'view': v.get('view'), 'box': v.get('box')})
    total = round(sum(final.values()), 2)
    r.update(model_indicators=model, model_total=round(sum(model.values()), 2), indicators=final, total=total,
             cls_rule='Low from classifier, otherwise from total score', score_mode=mode, comparison=cmp_)
    r['class'] = 'Low' if r['p_low'] >= 0.5 else to_class(max(total, 5))

    warns = []
    hi = [i for i in INDS if final[i] >= 1.5]
    if hi and r['class'] in ('Low', 'Moderate'):
        warns.append(f"Overall class is {r['class']}, but elevated indicator(s): " +
                     ', '.join(f"{i.replace('_', ' ')} ({final[i]:.1f}/3)" for i in hi) + '.')
    for c in cmp_:
        if c['status'] == 'disagree':
            warns.append(f"Model ({c['model']:.1f}) and visual check ({c['vlm']:g}) disagree on "
                         f"{c['indicator'].replace('_', ' ')}. Verify this one manually.")
    if not vlm:
        r['report'] = template_report(r)
    else:
        lines = [f"Overall risk: {r['class']} (score {total:.1f} / 18, {mode_text(mode)}).", '', vlm['summary'], '']
        if vlm['next_steps']:
            lines += ['Recommended next steps:'] + [f'{n}. {t}' for n, t in enumerate(vlm['next_steps'], 1)] + ['']
        lines += ['Screening aid only; not an engineering assessment.']
        r['report'] = '\n'.join(lines)
    r['warnings'] = warns; r['report_source'] = source
    return add_bangla(r, tr)

def predict(x):
    z = M['scaler'].transform(x)
    ind = {i: float(np.clip(M['reg_ind'][i].predict(z)[0], 0, 3)) for i in INDS}
    total = sum(ind.values()); p_low = float(M['clf'].predict_proba(z)[0, 1])
    cls = 'Low' if p_low >= 0.5 else to_class(max(total, 5))       # same rule as the notebook's hybrid_class
    return {'indicators': {k: round(v, 2) for k, v in ind.items()}, 'total': round(total, 2),
            'class': cls, 'p_low': round(p_low, 2)}

def ev(**k): return json.dumps(k) + '\n'

# ---------------------------------------------------------------- analysis (streams progress as JSON lines)
@app.post('/analyze')
async def analyze(facade: UploadFile = File(...), oblique45: UploadFile = File(...),
                  oblique60: UploadFile = File(...), nadir: UploadFile = File(...), name: str = Form('')):
    try:
        images = {}
        for v, f in zip(VIEWS, [facade, oblique45, oblique60, nadir]):
            im = Image.open(io.BytesIO(await f.read())); im.load(); images[v] = im
    except Exception:
        raise HTTPException(400, 'Could not read one of the images.')
    name = re.sub(r'\s+', ' ', name).strip()[:120]

    def run():
        try:
            yield ev(step='features', state='start', total=len(VIEWS))
            feats = []
            for n, v in enumerate(VIEWS, 1):
                feats.append(image_feat(images[v].convert('RGB')))
                yield ev(step='features', state='progress', n=n, total=len(VIEWS))
            yield ev(step='features', state='done')
            yield ev(step='model', state='start')
            r = predict(np.mean(feats, axis=0)[None])
            yield ev(step='model', state='done')
            yield ev(step='vlm', state='start')
            vlm, src = vlm_assess(images)
            yield ev(step='vlm', state='done' if vlm else 'skipped', source=src)
            yield ev(step='report', state='start')
            tr = translate_bn(vlm)
            r = storage.save(finish(r, vlm, src, tr), images, name)
            yield ev(step='report', state='done')
            if r['class'] != 'Low' and AUTO_TO:
                yield ev(step='email', state='start', to=AUTO_TO)
                r = auto_email(r)
                yield ev(step='email', state='done' if r['auto_email']['status'] == 'sent' else 'failed',
                         reason=r['auto_email']['status'], error=r['auto_email'].get('error', ''))
            else:
                reason = 'low' if r['class'] == 'Low' else 'off'
                r = storage.update(r['id'], auto_email={'status': reason, 'to': AUTO_TO}) or r
                yield ev(step='email', state='skipped', reason=reason)
            yield ev(result=r)
        except Exception as e:
            traceback.print_exc()
            yield ev(error=f'{type(e).__name__}: {e}')
    return StreamingResponse(run(), media_type='application/x-ndjson')

# ---------------------------------------------------------------- history
def _get(aid):
    r = storage.get(aid)
    if r is None: raise HTTPException(404, 'Analysis not found.')
    if 'report_bn' not in r and 'indicators' in r:     # analyses saved before Bangla support
        add_bangla(r)
    return r

@app.get('/history')
def history(): return storage.list_all()

@app.get('/history/{aid}')
def history_item(aid: str): return _get(aid)

@app.delete('/history/{aid}')
def history_delete(aid: str):
    _get(aid); storage.delete(aid); return {'ok': True}

@app.get('/photo/{aid}/{view}')
def photo(aid: str, view: str):
    p = storage.image_path(aid, view)
    if not p or not os.path.exists(p): raise HTTPException(404, 'Photo not found.')
    return FileResponse(p, media_type='image/jpeg', headers={'Cache-Control': 'max-age=86400'})

# ---------------------------------------------------------------- heatmap (on demand, cached in history)
@app.post('/heatmap/{aid}')
def heatmap(aid: str):
    r = _get(aid)
    def run():
        if r.get('heatmap'):
            yield ev(result=r['heatmap']); return
        try:
            import occlusion
            images = storage.load_images(aid)
            hm = None
            for msg in occlusion.heatmaps(images, M, grid=int(os.getenv('HEATMAP_GRID', '4'))):
                if 'result' in msg: hm = msg['result']
                else: yield ev(**msg)
            storage.update(aid, heatmap=hm)
            yield ev(result=hm)
        except Exception as e:
            traceback.print_exc()
            yield ev(error=f'{type(e).__name__}: {e}')
    return StreamingResponse(run(), media_type='application/x-ndjson')

# ---------------------------------------------------------------- PDF and email
def _pdf(r, lang='en'):
    import report_pdf
    return report_pdf.build_pdf(r, {v: storage.image_path(r['id'], v) for v in VIEWS}, lang)

def _pdf_name(r, lang='en'):
    base = re.sub(r'[^A-Za-z0-9]+', '_', r.get('name') or '').strip('_')[:40] or r['id']
    return f"BhobonAkkhi_{base}{'_bangla' if lang == 'bn' else ''}.pdf"

def _bn_ok():
    import report_pdf
    return report_pdf.bangla_available()

def _attachments(r, langs):
    """[(pdf bytes, filename)] for the requested languages; Bangla is skipped if its font setup is missing."""
    langs = [l for l in langs if l == 'en' or _bn_ok()] or ['en']
    return [(_pdf(r, l), _pdf_name(r, l)) for l in langs]

_auto = os.getenv('AUTO_EMAIL_TO', 'umme.simki@northsouth.edu').strip()
AUTO_TO = [] if _auto.lower() in ('', 'off', 'none', 'no', '0', 'false') else \
    [a for a in re.split(r'[,;\s]+', _auto) if a]
AUTO_LANGS = {'bn': ['bn'], 'both': ['en', 'bn']}.get(os.getenv('AUTO_EMAIL_LANG', 'en').strip().lower(), ['en'])

def auto_email(r):
    """Email a non-Low report to AUTO_EMAIL_TO. Never raises: the outcome is saved in r['auto_email']."""
    import mailer
    info = {'to': AUTO_TO, 'at': time.time()}
    try:
        mailer.send_report(AUTO_TO, _attachments(r, AUTO_LANGS), r, auto=True)
        info['status'] = 'sent'
    except mailer.NotConfigured as e:
        info.update(status='not_configured', error=str(e))
    except Exception as e:
        traceback.print_exc()
        info.update(status='failed', error=f'{type(e).__name__}: {e}'[:300])
    return storage.update(r['id'], auto_email=info) or {**r, 'auto_email': info}

@app.get('/report/{aid}.pdf')
def report_pdf_file(aid: str, download: int = 0, lang: str = 'en'):
    r = _get(aid)
    lang = 'bn' if lang == 'bn' else 'en'
    if lang == 'bn' and not _bn_ok():
        import report_pdf
        raise HTTPException(503, 'Bangla PDF is not set up: run "pip install uharfbuzz" and put HindSiliguri-Regular.ttf '
                                 f'and HindSiliguri-Bold.ttf in the fonts folder. ({report_pdf.bangla_error()})')
    disp = 'attachment' if download else 'inline'
    return Response(_pdf(r, lang), media_type='application/pdf',
                    headers={'Content-Disposition': f'{disp}; filename="{_pdf_name(r, lang)}"'})

@app.post('/email/{aid}')
def email(aid: str, payload: dict = Body(...)):
    import mailer
    r = _get(aid)
    to = mailer.parse_recipients(str(payload.get('to', '')))
    if not to: raise HTTPException(400, 'Enter a valid email address (up to 5, separated by commas).')
    try:
        lang = 'bn' if payload.get('lang') == 'bn' else 'en'
        mailer.send_report(to, _attachments(r, [lang]), r, note=str(payload.get('note', ''))[:2000])
    except mailer.NotConfigured as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(502, f'Could not send the email: {e}')
    return {'ok': True, 'to': to}

@app.get('/config')
def config():
    import mailer
    return {'email': mailer.configured(), 'score_mode': SCORE_MODE, 'auto_email_to': AUTO_TO, 'bn_pdf': _bn_ok(),
            'vlm': os.getenv('REPORT_BACKEND') or ('gemini' if os.getenv('GEMINI_API_KEY') else 'template')}

@app.get('/')
def index(): return FileResponse(os.path.join(BASE, 'index.html'))