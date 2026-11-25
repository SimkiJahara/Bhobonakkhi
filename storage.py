"""History: every analysis is saved in data/history.db, with its four photos in data/photos/<id>/."""
import os, re, json, time, uuid, sqlite3, threading
from PIL import Image

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, 'data')
PHOTOS = os.path.join(DATA, 'photos')
DB = os.path.join(DATA, 'history.db')
VIEWS = ['facade', 'oblique45', 'oblique60', 'nadir']
_ID = re.compile(r'^[0-9a-f]{12}$')
_lock = threading.Lock()

def _db():
    os.makedirs(PHOTOS, exist_ok=True)
    c = sqlite3.connect(DB)
    c.execute('CREATE TABLE IF NOT EXISTS analyses (id TEXT PRIMARY KEY, created REAL, name TEXT, '
              'class TEXT, total REAL, result TEXT)')
    return c

def _ok(aid): return isinstance(aid, str) and bool(_ID.match(aid))

def image_path(aid, view):
    if not _ok(aid) or view not in VIEWS: return None
    return os.path.join(PHOTOS, aid, f'{view}.jpg')

def save(result, images, name=''):
    """Store the result and a copy of the photos (longest side 1600 px). Adds id, name and created to result."""
    aid = uuid.uuid4().hex[:12]
    os.makedirs(os.path.join(PHOTOS, aid), exist_ok=True)
    for v in VIEWS:
        im = images[v].convert('RGB'); im.thumbnail((1600, 1600))
        im.save(image_path(aid, v), 'JPEG', quality=88)      # saved without EXIF so it shows the way the models saw it
    result.update(id=aid, name=name or '', created=time.time())
    with _lock, _db() as c:
        c.execute('INSERT INTO analyses VALUES (?,?,?,?,?,?)',
                  (aid, result['created'], result['name'], result['class'], result['total'], json.dumps(result)))
    return result

def get(aid):
    if not _ok(aid): return None
    with _db() as c:
        row = c.execute('SELECT result FROM analyses WHERE id=?', (aid,)).fetchone()
    return json.loads(row[0]) if row else None

def update(aid, **fields):
    with _lock:
        r = get(aid)
        if r is None: return None
        r.update(fields)
        with _db() as c:
            c.execute('UPDATE analyses SET result=? WHERE id=?', (json.dumps(r), aid))
    return r

def list_all(limit=500):
    with _db() as c:
        rows = c.execute('SELECT result FROM analyses ORDER BY created DESC LIMIT ?', (limit,)).fetchall()
    out = []
    for (txt,) in rows:
        r = json.loads(txt)
        out.append({k: r.get(k) for k in ('id', 'name', 'created', 'class', 'total', 'indicators',
                                           'model_indicators', 'score_mode', 'report_source')})
    return out

def delete(aid):
    if not _ok(aid): return
    with _lock, _db() as c:
        c.execute('DELETE FROM analyses WHERE id=?', (aid,))
    folder = os.path.join(PHOTOS, aid)
    if os.path.isdir(folder):
        for f in os.listdir(folder): os.remove(os.path.join(folder, f))
        os.rmdir(folder)

def load_images(aid):
    return {v: Image.open(image_path(aid, v)).convert('RGB') for v in VIEWS}