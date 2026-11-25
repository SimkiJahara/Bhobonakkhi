"""Send the PDF report by email through Gmail (or any SMTP server).
.env:  SMTP_USER=mave.quake.cse499@gmail.com
       SMTP_PASSWORD=<16-letter Gmail app password>
Optional: SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 465 = SSL; 587 = STARTTLS).
Reports that are not Low risk are sent automatically by app.py (see AUTO_EMAIL_TO there)."""
import os, re, ssl, smtplib, datetime
from email.message import EmailMessage

class NotConfigured(RuntimeError): pass

_EMAIL = re.compile(r'^[^@\s,;<>"]+@[^@\s,;<>"]+\.[A-Za-z]{2,}$')

def configured():
    return bool(os.getenv('SMTP_USER') and os.getenv('SMTP_PASSWORD'))

def parse_recipients(text, limit=5):
    parts = [p.strip() for p in re.split(r'[,;\s]+', text or '') if p.strip()]
    if not parts or len(parts) > limit or not all(_EMAIL.match(p) for p in parts): return []
    return parts

def send_report(to, attachments, r, note='', auto=False):
    """attachments: list of (pdf_bytes, filename). auto=True adds a line saying why it was sent automatically."""
    user, pw = os.getenv('SMTP_USER'), (os.getenv('SMTP_PASSWORD') or '').replace(' ', '')
    if not user or not pw:
        raise NotConfigured('Email is not set up yet. Add SMTP_USER and SMTP_PASSWORD to the .env file and restart the server.')
    host, port = os.getenv('SMTP_HOST', 'smtp.gmail.com'), int(os.getenv('SMTP_PORT', '465'))
    name = r.get('name') or f"building {r['id']}"
    when = datetime.datetime.fromtimestamp(r.get('created') or 0).strftime('%d %B %Y')

    msg = EmailMessage()
    msg['Subject'] = ('[Automatic] ' if auto else '') + f"BhobonAkkhi screening report: {name} ({r['class']} risk)"
    msg['From'] = f'BhobonAkkhi <{user}>'
    msg['To'] = ', '.join(to)
    body = [f'Hello,', '', f'Attached is the BhobonAkkhi seismic screening report for {name}, analysed on {when}.', '',
            f"Overall risk: {r['class']} (score {r['total']:.1f} out of 18)"]
    if auto:
        body += ['', f"This email was sent automatically because the building was rated {r['class']} risk. "
                     'Every analysis above Low risk is sent to this address.']
    if len(attachments) > 1:
        body += ['', 'The report is attached in English and in Bangla.']
    if note.strip():
        body += ['', 'Message from the sender:', note.strip()]
    body += ['', 'This is an automated screening aid based on drone photos, not a structural engineering assessment.',
             'A qualified engineer should inspect the building before any decision is made.', '', 'BhobonAkkhi']
    msg.set_content('\n'.join(body))
    for pdf_bytes, filename in attachments:
        msg.add_attachment(pdf_bytes, maintype='application', subtype='pdf', filename=filename)

    ctx = ssl.create_default_context()
    if port == 465:
        with smtplib.SMTP_SSL(host, port, context=ctx, timeout=30) as s:
            s.login(user, pw); s.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=30) as s:
            s.starttls(context=ctx); s.login(user, pw); s.send_message(msg)