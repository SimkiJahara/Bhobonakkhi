"""Build the PDF report (English or Bangla) with reportlab: verdict, annotated photos, radar chart, indicator table,
report text.

Bangla needs two things (both free):
  - pip install uharfbuzz          (ReportLab uses it to join Bengali letters correctly)
  - fonts/HindSiliguri-Regular.ttf and fonts/HindSiliguri-Bold.ttf next to this file
    (Hind Siliguri has both Bengali and Latin letters, so one font covers the whole page)."""
import io, os, re, math, datetime
from xml.sax.saxutils import escape
from PIL import Image, ImageDraw, ImageFont, features as pil_features
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage,
                                KeepTogether, ListFlowable, ListItem, Flowable)
from reportlab.graphics.shapes import Drawing, Polygon, Line, Rect
from reportlab.graphics import renderPDF

BASE = os.path.dirname(os.path.abspath(__file__))
FONT_DIR = os.path.join(BASE, 'fonts')
BN_REG, BN_BOLD = os.path.join(FONT_DIR, 'HindSiliguri-Regular.ttf'), os.path.join(FONT_DIR, 'HindSiliguri-Bold.ttf')

VIEWS = ['facade', 'oblique45', 'oblique60', 'nadir']
INDS = ['Soft_Story', 'Cracks', 'Plan_Irregularity', 'Material_Degradation', 'Pounding', 'Overhang']
IND_COLOR = {'Soft_Story': '#d81b60', 'Cracks': '#e53935', 'Plan_Irregularity': '#8e24aa',
             'Material_Degradation': '#f57c00', 'Pounding': '#1e88e5', 'Overhang': '#00897b'}
CLS_COLOR = {'Low': '#1f7a48', 'Moderate': '#935f00', 'High': '#b3430c', 'Critical': '#b3202b'}
INK, MUT, RULE, BP, AI = colors.HexColor('#17211f'), colors.HexColor('#5d6b66'), colors.HexColor('#cdd2ca'), \
    colors.HexColor('#1d3f8f'), colors.HexColor('#d9822b')

# ------------------------------------------------------------------ Bangla font setup
_BN_ERR = None
try:
    import uharfbuzz  # noqa: F401  (needed by ReportLab's text shaping)
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    if not (os.path.exists(BN_REG) and os.path.exists(BN_BOLD)):
        raise FileNotFoundError(f'Bangla font files not found in {FONT_DIR}')
    pdfmetrics.registerFont(TTFont('BnReg', BN_REG))
    pdfmetrics.registerFont(TTFont('BnBold', BN_BOLD))
    pdfmetrics.registerFontFamily('BnReg', normal='BnReg', bold='BnBold', italic='BnReg', boldItalic='BnBold')
except Exception as e:  # missing uharfbuzz or font files: English PDF still works
    _BN_ERR = f'{type(e).__name__}: {e}'

def bangla_available(): return _BN_ERR is None
def bangla_error(): return _BN_ERR

BN_DIGITS = str.maketrans('0123456789', '০১২৩৪৫৬৭৮৯')
BN_MONTHS = ['জানুয়ারি', 'ফেব্রুয়ারি', 'মার্চ', 'এপ্রিল', 'মে', 'জুন', 'জুলাই', 'আগস্ট', 'সেপ্টেম্বর', 'অক্টোবর',
             'নভেম্বর', 'ডিসেম্বর']

# ------------------------------------------------------------------ all fixed text, per language
TXT = {
 'en': dict(
    title='BhobonAkkhi · Seismic screening report', unnamed='Unnamed building', id='ID',
    cls={'Low': 'Low risk', 'Moderate': 'Moderate risk', 'High': 'High risk', 'Critical': 'Critical risk'},
    of='out of 18', final='Final score', mode={'average': 'Average of the model and the AI visual check',
                                               'model': 'BhobonAkkhi model'},
    photos='Photos', boxnote='Coloured boxes mark where the AI visual check saw the evidence for each indicator.',
    vn={'facade': 'Façade', 'oblique45': 'Oblique 45°', 'oblique60': 'Oblique 60°', 'nadir': 'Nadir'},
    rad='Model vs AI visual check', lgM='Model', lgA='AI visual check', lgF='Final (average)',
    head=['Indicator', 'Model', 'AI check', 'Final', 'Agreement', 'What the AI saw'],
    status={'exact': 'Same', 'close': 'Within 1', 'disagree': 'Differs', 'no evidence': 'Not visible'},
    scores='Indicator scores (0 = no concern, 3 = severe)', rep='Written report',
    src={'gemini': 'Written by Gemini from the four photos.', 'qwen': 'Written by Qwen from the four photos.',
         'template': 'Fixed template text; no AI model wrote this report.'},
    foot='BhobonAkkhi · Screening aid only, not a structural engineering assessment.', page='Page',
    ind={'Soft_Story': 'Soft story', 'Cracks': 'Cracks', 'Plan_Irregularity': 'Plan irregularity',
         'Material_Degradation': 'Material degradation', 'Pounding': 'Pounding', 'Overhang': 'Overhang'},
    short={'Material_Degradation': 'Material deg.', 'Plan_Irregularity': 'Plan irreg.'}),
 'bn': dict(
    title='ভবনআখি · ভূমিকম্প ঝুঁকি স্ক্রিনিং প্রতিবেদন', unnamed='নামহীন ভবন', id='আইডি',
    cls={'Low': 'নিম্ন ঝুঁকি', 'Moderate': 'মাঝারি ঝুঁকি', 'High': 'উচ্চ ঝুঁকি', 'Critical': 'অতি উচ্চ ঝুঁকি'},
    of='১৮-এর মধ্যে', final='চূড়ান্ত স্কোর', mode={'average': 'মডেল ও এআই চাক্ষুষ যাচাইয়ের গড়',
                                                  'model': 'ভবনআখি মডেল'},
    photos='ছবি', boxnote='রঙিন বাক্সগুলো দেখায় এআই চাক্ষুষ যাচাই প্রতিটি সূচকের প্রমাণ ছবির কোথায় দেখেছে।',
    vn={'facade': 'সম্মুখভাগ', 'oblique45': 'তির্যক ৪৫°', 'oblique60': 'তির্যক ৬০°', 'nadir': 'উপর থেকে'},
    rad='মডেল বনাম এআই চাক্ষুষ যাচাই', lgM='মডেল', lgA='এআই চাক্ষুষ যাচাই', lgF='চূড়ান্ত (গড়)',
    head=['সূচক', 'মডেল', 'এআই', 'চূড়ান্ত', 'মিল', 'এআই যা দেখেছে'],
    status={'exact': 'একই', 'close': '১-এর মধ্যে', 'disagree': 'পার্থক্য', 'no evidence': 'দেখা যায়নি'},
    scores='সূচকের স্কোর (০ = কোনো উদ্বেগ নেই, ৩ = গুরুতর)', rep='লিখিত প্রতিবেদন',
    src={'gemini': 'চারটি ছবি থেকে Gemini এই প্রতিবেদন লিখেছে।', 'qwen': 'চারটি ছবি থেকে Qwen এই প্রতিবেদন লিখেছে।',
         'template': 'এটি নির্দিষ্ট টেমপ্লেটের লেখা; কোনো এআই মডেল লেখেনি।'},
    foot='ভবনআখি · শুধু স্ক্রিনিং সহায়ক, কাঠামোগত প্রকৌশল মূল্যায়ন নয়।', page='পৃষ্ঠা',
    ind={'Soft_Story': 'সফট স্টোরি', 'Cracks': 'ফাটল', 'Plan_Irregularity': 'নকশার অনিয়ম',
         'Material_Degradation': 'উপকরণের ক্ষয়', 'Pounding': 'পাউন্ডিং', 'Overhang': 'ওভারহ্যাং'},
    short={}),
}

def _styles(lang):
    bn = lang == 'bn'
    reg, bold = ('BnReg', 'BnBold') if bn else ('Helvetica', 'Helvetica-Bold')
    k = dict(shaping=1) if bn else {}
    lead = 1.15 if bn else 1.0           # Bengali vowel signs need a little more line height
    ps = lambda name, **kw: ParagraphStyle(name, **{**kw, **k})
    return {
     'h1': ps('h1', fontName=bold, fontSize=20, leading=24 * lead, textColor=BP),
     'h2': ps('h2', fontName=bold, fontSize=13, leading=16 * lead, textColor=INK, spaceBefore=10, spaceAfter=5),
     'p': ps('p', fontName=reg, fontSize=10, leading=14 * lead, textColor=INK),
     'mut': ps('mut', fontName=reg, fontSize=9, leading=12 * lead, textColor=MUT),
     'cell': ps('cell', fontName=reg, fontSize=8.5, leading=11 * lead, textColor=INK),
     'big': ps('big', fontName=bold, fontSize=26, leading=30 * lead),
     'lab': ps('lab', fontName=reg, fontSize=7.5, leading=9 * lead, textColor=MUT),
     'foot': ps('foot', fontName=reg, fontSize=7.5, leading=9 * lead, textColor=MUT),
     'warn': ps('warn', fontName=reg, fontSize=9.5, leading=13 * lead, textColor=colors.HexColor('#6b4500'),
                backColor=colors.HexColor('#fbefd5'), borderPadding=(5, 6, 5, 6), spaceAfter=8, leftIndent=2),
     '_reg': reg, '_bold': bold,
    }

def annotated(path, comparison, width_mm, label_of, bn=False):
    """Photo with the AI's evidence boxes drawn on it."""
    im = Image.open(path).convert('RGB'); im.thumbnail((900, 900))
    W, H = im.size; d = ImageDraw.Draw(im); lw = max(2, W // 220)
    size = max(14, W // 34)
    # Bengali labels on the photo need Pillow built with raqm (text shaping); otherwise keep English labels
    use_bn = bn and pil_features.check('raqm') and os.path.exists(BN_REG)
    try:
        font = ImageFont.truetype(BN_REG, size, layout_engine=ImageFont.Layout.RAQM) if use_bn \
            else ImageFont.load_default(size=size)
    except Exception:
        use_bn = False
        try: font = ImageFont.load_default(size=size)
        except TypeError: font = ImageFont.load_default()
    view = path.replace('\\', '/').rsplit('/', 1)[-1].split('.')[0]
    for c in comparison or []:
        if c.get('view') != view or not c.get('box'): continue
        y0, x0, y1, x1 = c['box']; box = (x0 * W / 1000, y0 * H / 1000, x1 * W / 1000, y1 * H / 1000)
        col = IND_COLOR.get(c['indicator'], '#ffffff')
        d.rectangle(box, outline=col, width=lw)
        label = label_of(c['indicator'], use_bn)
        tb = d.textbbox((0, 0), label, font=font); tw, th = tb[2] - tb[0], tb[3] - tb[1]
        ty = box[1] - th - 6 if box[1] - th - 6 > 0 else box[1]
        d.rectangle((box[0], ty, box[0] + tw + 8, ty + th + 6), fill=col)
        d.text((box[0] + 4, ty + 2 - tb[1]), label, fill='white', font=font)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=85); buf.seek(0)
    w = width_mm * mm
    return RLImage(buf, width=w, height=w * H / W)

class Radar(Flowable):
    """Radar chart. Axis labels are drawn as small Paragraphs so Bengali letters join correctly."""
    def __init__(self, series, labels, style, size=74 * mm):
        super().__init__()
        self.series, self.labels, self.style, self.size = series, labels, style, size
        self.width, self.height = size + 110, size + 30

    def wrap(self, aw, ah): return self.width, self.height

    def draw(self):
        size = self.size
        cx, cy, R = self.width / 2, self.height / 2, size / 2 - 6
        ang = [math.pi / 2 - 2 * math.pi * k / len(INDS) for k in range(len(INDS))]
        pt = lambda k, v: (cx + R * v / 3 * math.cos(ang[k]), cy + R * v / 3 * math.sin(ang[k]))
        d = Drawing(self.width, self.height)
        for lvl in (1, 2, 3):
            d.add(Polygon([c for k in range(len(INDS)) for c in pt(k, lvl)], fillColor=None, strokeColor=RULE,
                          strokeWidth=0.6))
        for k in range(len(INDS)):
            x, y = pt(k, 3); d.add(Line(cx, cy, x, y, strokeColor=RULE, strokeWidth=0.6))
        for _, vals, col, filled, dashed in self.series:
            pts = [c for k, ind in enumerate(INDS) for c in pt(k, max(0, min(3, vals.get(ind) or 0)))]
            fc = colors.Color(col.red, col.green, col.blue, alpha=0.15) if filled else None
            d.add(Polygon(pts, fillColor=fc, strokeColor=col, strokeWidth=2.2 if dashed else 1.6,
                          strokeDashArray=[4, 3] if dashed else None))
        renderPDF.draw(d, self.canv, 0, 0)
        lw = 60
        for k, ind in enumerate(INDS):
            lx, ly = pt(k, 3.35); c = math.cos(ang[k])
            align, x = (TA_CENTER, lx - lw / 2) if abs(c) < 0.2 else ((TA_LEFT, lx) if c > 0 else (TA_RIGHT, lx - lw))
            p = Paragraph(escape(self.labels[ind]), ParagraphStyle('rl', parent=self.style, alignment=align))
            _, h = p.wrap(lw, 40)
            yy = ly - h / 2 if abs(c) >= 0.2 else (ly if math.sin(ang[k]) > 0 else ly - h)
            p.drawOn(self.canv, x, yy)

def build_pdf(r, photos, lang='en'):
    lang = 'bn' if lang == 'bn' else 'en'
    if lang == 'bn' and not bangla_available():
        raise RuntimeError('Bangla PDF is not available: ' + (_BN_ERR or ''))
    bn, L, S = lang == 'bn', TXT[lang], _styles(lang)
    N = (lambda x: str(x).translate(BN_DIGITS)) if bn else (lambda x: str(x))
    P = lambda text, style='p': Paragraph(escape(str(text)), S[style])
    nm = lambda k: L['ind'].get(k, k.replace('_', ' ').capitalize())

    def label_of(k, bengali_ok):            # labels drawn on photos
        return nm(k) if (not bn or bengali_ok) else TXT['en']['ind'][k]

    buf = io.BytesIO()
    dt = datetime.datetime.fromtimestamp(r.get('created') or 0)
    created = (f'{N(dt.day)} {BN_MONTHS[dt.month - 1]} {N(dt.year)}, {N(dt.strftime("%H:%M"))}' if bn
               else dt.strftime('%d %B %Y, %H:%M'))

    def frame(c, doc):
        c.saveState()
        ft = Paragraph(escape(L['foot']), S['foot']); ft.wrap(130 * mm, 20); ft.drawOn(c, 18 * mm, 8.5 * mm)
        pg = Paragraph(escape(f"{L['page']} {N(doc.page)}"), ParagraphStyle('pg', parent=S['foot'], alignment=TA_RIGHT))
        pg.wrap(40 * mm, 20); pg.drawOn(c, A4[0] - 18 * mm - 40 * mm, 8.5 * mm)
        c.setFillColor(BP); c.rect(0, A4[1] - 6 * mm, A4[0], 6 * mm, stroke=0, fill=1); c.restoreState()

    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=18 * mm, title=f"BhobonAkkhi report {r.get('name') or r['id']}",
                            author='BhobonAkkhi', lang='bn-BD' if bn else 'en')
    st = []
    st += [Paragraph(escape(L['title']), S['h1']), Spacer(1, 3),
           P(f"{r.get('name') or L['unnamed']}  ·  {created}  ·  {L['id']} {r['id']}", 'mut'), Spacer(1, 8)]

    col = colors.HexColor(CLS_COLOR.get(r['class'], '#17211f'))
    mode = L['mode']['average' if r.get('score_mode') == 'average' else 'model']
    verdict = Table([[Paragraph(f"<font color='{CLS_COLOR.get(r['class'], '#17211f')}'>"
                                f"{escape(L['cls'].get(r['class'], r['class']))}</font>", S['big']),
                      Paragraph(f"<b><font size=20>{N(format(r['total'], '.1f'))}</font></b> {escape(L['of'])}<br/>"
                                f"<font size=8 color='#5d6b66'>{escape(L['final'])}: {escape(mode)}</font>", S['p'])]],
                    colWidths=[90 * mm, 84 * mm])
    verdict.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LINEBELOW', (0, 0), (-1, 0), 2, col),
                                 ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
    st += [verdict, Spacer(1, 8)]
    for w in ((bn and r.get('warnings_bn')) or r.get('warnings') or []): st.append(P(w, 'warn'))

    # photos 2 x 2
    cells = [[annotated(photos[v], r.get('comparison'), 82, label_of, bn), P(L['vn'][v], 'mut')] for v in VIEWS]
    ph = Table([[cells[0], cells[1]], [cells[2], cells[3]]], colWidths=[87 * mm, 87 * mm])
    ph.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LEFTPADDING', (0, 0), (-1, -1), 0)]))
    has_box = any(c.get('box') for c in r.get('comparison') or [])
    st += [Paragraph(escape(L['photos']), S['h2'])]
    if has_box: st.append(P(L['boxnote'], 'mut'))
    st += [Spacer(1, 4), ph]
    if has_box:   # colour key, so the indicator names are readable in either language
        boxed = [i for i in INDS if any(c.get('box') and c['indicator'] == i for c in r['comparison'])]
        key = []
        for i in boxed:
            dr = Drawing(10, 8); dr.add(Rect(0, 1, 10, 6, fillColor=colors.HexColor(IND_COLOR[i]), strokeColor=None))
            key += [dr, P(nm(i), 'mut')]
        kt = Table([key], colWidths=[4 * mm, 25 * mm] * len(boxed), hAlign='LEFT')
        kt.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 1)]))
        st += [Spacer(1, 4), kt]

    # radar + legend
    series = [(L['lgM'], r.get('model_indicators') or r['indicators'], BP, True, False)]
    if r.get('comparison'):
        series.append((L['lgA'], {c['indicator']: c['vlm'] for c in r['comparison']}, AI, True, False))
        if r.get('score_mode') == 'average': series.append((L['lgF'], r['indicators'], INK, False, True))
    labels = {i: L['short'].get(i, nm(i)) for i in INDS}
    cells = []
    for label, _, c, _, _ in series:
        dr = Drawing(10, 8); dr.add(Rect(0, 1, 10, 6, fillColor=c, strokeColor=None)); cells += [dr, P(label, 'mut')]
    lg = Table([cells], colWidths=[4 * mm, 34 * mm] * len(series))
    lg.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('LEFTPADDING', (0, 0), (-1, -1), 1)]))
    st += [KeepTogether([Paragraph(escape(L['rad']), S['h2']), Radar(series, labels, S['lab']), Spacer(1, 8), lg])]

    # indicator table
    cm = {c['indicator']: c for c in r.get('comparison') or []}
    rows = [[Paragraph(f'<b>{escape(h)}</b>', S['cell']) for h in L['head']]]
    mi = r.get('model_indicators') or r['indicators']
    for i in INDS:
        c = cm.get(i, {}); vs = c.get('vlm')
        ev_txt = (bn and c.get('evidence_bn')) or c.get('evidence') or ''
        rows.append([P(nm(i), 'cell'), P(N(f'{mi[i]:.1f}'), 'cell'), P('–' if vs is None else N(f'{vs:g}'), 'cell'),
                     Paragraph(f"<b>{N(format(r['indicators'][i], '.1f'))}</b>", S['cell']),
                     P(L['status'].get(c.get('status'), '–'), 'cell'), P(ev_txt, 'cell')])
    t = Table(rows, colWidths=[32 * mm, 13 * mm, 15 * mm, 13 * mm, 21 * mm, 80 * mm], repeatRows=1)
    t.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, 0), 1, INK), ('LINEBELOW', (0, 1), (-1, -1), 0.4, RULE),
                           ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 4),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]))
    st += [Paragraph(escape(L['scores']), S['h2']), t]

    st += [Paragraph(escape(L['rep']), S['h2']), P(L['src'].get(r.get('report_source'), ''), 'mut'), Spacer(1, 4)]
    st += md_flow((bn and r.get('report_bn')) or r.get('report'), S, N)
    doc.build(st, onFirstPage=frame, onLaterPages=frame)
    return buf.getvalue()

def md_flow(text, S, N=str):
    """Very small markdown: '- ' bullets, '1. ' numbers, '**bold**', '# heading', paragraphs."""
    out, items, kind = [], [], None
    def flush():
        nonlocal items, kind
        if items:
            ol = kind == 'ol'
            out.append(ListFlowable([ListItem(Paragraph(t, S['p']), leftIndent=14) for t in items],
                                    bulletType='1' if ol else 'bullet', start=1 if ol else None,
                                    bulletFormat=(lambda v: f'{N(v)}.') if ol else None,
                                    bulletFontName=S['_reg'], leftIndent=14, bulletFontSize=8 if not ol else 10))
            out.append(Spacer(1, 6))
        items, kind = [], None
    inl = lambda t: re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', escape(t))
    for raw in (text or '').split('\n'):
        l = raw.strip()
        if not l or re.match(r'^-{3,}$', l): flush(); continue
        m = re.match(r'^[*-]\s+(.*)', l); n = re.match(r'^[0-9০-৯]+\.\s+(.*)', l); h = re.match(r'^#{1,6}\s+(.*)', l)
        if m:
            if kind != 'ul': flush(); kind = 'ul'
            items.append(inl(m.group(1)))
        elif n:
            if kind != 'ol': flush(); kind = 'ol'
            items.append(inl(n.group(1)))
        else:
            flush(); out.append(Paragraph(inl(h.group(1) if h else l), S['h2'] if h else S['p'])); out.append(Spacer(1, 4))
    flush()
    return out