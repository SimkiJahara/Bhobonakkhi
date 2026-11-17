"""Capstone evaluation: how good is the independent visual check, the ridge model, and the two combined?

Usage:  python evaluate_vlm.py /path/to/All /path/to/labels.csv
 - runs the blind Gemini check on every labelled building (original photos only)
 - results are cached in results/vlm_cache.jsonl, so you can stop and resume (free-tier limits!)
 - compares against your labels: VLM alone | ridge model (5-fold cross-validated) | average of both
Needs GEMINI_API_KEY (in .env)."""
import os, re, sys, json, time, warnings
import numpy as np, pandas as pd
from PIL import Image
from scipy.stats import spearmanr
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from sklearn.metrics import cohen_kappa_score, roc_auc_score, mean_absolute_error, accuracy_score
warnings.filterwarnings('ignore')

OUT = 'results'
def to_class(t): return 'Low' if t < 5 else 'Moderate' if t < 10 else 'High' if t < 14 else 'Critical'

def view_of(f):
    f = f.lower()
    if 'nadir' in f: return 'nadir'
    if 'facade' in f or 'fa\u00e7ade' in f: return 'facade'
    if re.search(r'oblique\D{0,2}45', f): return 'oblique45'
    if re.search(r'oblique\D{0,2}60', f): return 'oblique60'

def load_labels(path, INDS):
    raw = pd.read_csv(path); raw.columns = [c.strip() for c in raw.columns]
    for ind in INDS:
        for c in raw.columns:
            if c.lower().startswith(ind.lower()[:8]) and c != ind: raw = raw.rename(columns={c: ind})
    raw['Building'] = raw['Building'].astype(str).str.strip().str.upper()
    return raw.drop_duplicates('Building').reset_index(drop=True)

def get_images(root, b):
    d = os.path.join(root, b)
    return {view_of(f): Image.open(os.path.join(d, f)) for f in sorted(os.listdir(d))
            if view_of(f) and f.lower().endswith(('.jpg', '.jpeg', '.png'))}

def run_vlm(buildings, root, gemini_raw, parse_json, INDS):
    os.makedirs(OUT, exist_ok=True); cache_f = os.path.join(OUT, 'vlm_cache.jsonl'); cache = {}
    if os.path.exists(cache_f):
        for line in open(cache_f): r = json.loads(line); cache[r['building']] = r['vlm']
    with open(cache_f, 'a') as f:
        for k, b in enumerate(buildings, 1):
            if b in cache: continue
            for attempt in range(5):
                try:
                    cache[b] = parse_json(gemini_raw(get_images(root, b))); break
                except Exception as e:
                    wait = 10 * 2 ** attempt; print(f'  {b}: {str(e)[:90]} -> retry in {wait}s'); time.sleep(wait)
            else:
                print(f'  {b}: giving up'); continue
            f.write(json.dumps({'building': b, 'vlm': cache[b]}) + '\n'); f.flush()
            print(f'[{k}/{len(buildings)}] {b} done'); time.sleep(4)
    S = np.full((len(buildings), len(INDS)), np.nan)
    for i, b in enumerate(buildings):
        for j, ind in enumerate(INDS):
            try: S[i, j] = float(cache[b]['indicators'][ind]['score'])
            except Exception: pass
    return S

def oof_model(X, T, seeds=(0, 1, 2)):
    P = np.zeros((len(seeds),) + T.shape)
    for si, s in enumerate(seeds):
        for tr, te in KFold(5, shuffle=True, random_state=s).split(X):
            for j in range(T.shape[1]):
                m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13))).fit(X[tr], T[tr, j])
                P[si, te, j] = m.predict(X[te])
    return np.clip(P.mean(0), 0, 3)

def ind_table(T, setups, INDS):
    rows = []
    for name, P in setups.items():
        for j, ind in enumerate(INDS):
            ok = ~np.isnan(P[:, j]); y, p = T[ok, j], P[ok, j]; pr = np.clip(np.round(p), 0, 3).astype(int)
            rows.append(dict(setup=name, indicator=ind, n=int(ok.sum()), abstained=int((~ok).sum()),
                MAE=mean_absolute_error(y, p), spearman=spearmanr(p, y)[0], exact=(pr == y).mean(),
                within1=(abs(pr - y) <= 1).mean(),
                kappa_w=cohen_kappa_score(y.astype(int), pr, weights='quadratic', labels=[0, 1, 2, 3])))
    return pd.DataFrame(rows).round(3)

def total_table(T, setups):
    y = T.sum(1); low = (y < 5).astype(int); true4 = np.array([to_class(v) for v in y]); rows = []
    for name, P in setups.items():
        ok = ~np.isnan(P).any(1); s = P[ok].sum(1); c4 = np.array([to_class(v) for v in s])
        rows.append(dict(setup=name, n=int(ok.sum()), MAE_total=mean_absolute_error(y[ok], s),
            spearman=spearmanr(s, y[ok])[0], acc_4class=accuracy_score(true4[ok], c4),
            within_one_class=np.mean([abs(['Low', 'Moderate', 'High', 'Critical'].index(a) - ['Low', 'Moderate', 'High', 'Critical'].index(b)) <= 1
                                      for a, b in zip(true4[ok], c4)]),
            AUC_low_vs_rest=roc_auc_score(low[ok], -s) if 0 < low[ok].sum() < ok.sum() else np.nan))
    return pd.DataFrame(rows).round(3)

def main():
    from app import gemini_raw, parse_json, INDS          # also loads .env
    from features import building_feat
    root, labels = sys.argv[1], sys.argv[2]
    df = load_labels(labels, INDS); T = df[INDS].values.astype(float); B = list(df['Building'])
    S = run_vlm(B, root, gemini_raw, parse_json, INDS)
    ff = os.path.join(OUT, 'features_cache.npy')
    if os.path.exists(ff): X = np.load(ff)
    else:
        X = np.array([building_feat(get_images(root, b)) for b in B]); np.save(ff, X)
    M = oof_model(X, T)
    H = np.where(np.isnan(S), M, (S + M) / 2)           # average; fall back to model if VLM abstained
    setups = {'VLM alone (blind)': S, 'Ridge model (CV)': M, 'Average of both': H}
    t_ind, t_tot = ind_table(T, setups, INDS), total_table(T, setups)
    print('\n=== PER INDICATOR ==='); print(t_ind.to_string(index=False))
    print('\n=== TOTAL SCORE / CLASS ==='); print(t_tot.to_string(index=False))
    t_ind.to_csv(os.path.join(OUT, 'per_indicator.csv'), index=False); t_tot.to_csv(os.path.join(OUT, 'total.csv'), index=False)
    pb = pd.DataFrame({'Building': B}); 
    for j, ind in enumerate(INDS): pb[f'{ind}_true'] = T[:, j]; pb[f'{ind}_vlm'] = S[:, j]; pb[f'{ind}_model'] = M[:, j].round(2)
    pb.to_csv(os.path.join(OUT, 'per_building.csv'), index=False); print('\nSaved CSVs in', OUT)

if __name__ == '__main__':
    main()