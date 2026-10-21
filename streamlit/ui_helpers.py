"""
streamlit/ui_helpers.py  -  everything the web UI needs that is NOT search logic
================================================================================

Sections
    1. PDF -> profile parsing   : rule-based (no LLM, no API cost) extraction of
                                  name, title, location, years, skills, software,
                                  certifications, education and experience so that
                                  EVERY search result can show real details.
    2. Pictures                 : generated SVG avatars, company badges, hero banner
                                  (no image files or downloads needed).
    3. Matching helpers         : resolve search results to profiles, skill overlap.
    4. Offline search           : BM25-only fallback so the UI works without API keys.
    5. HTML/CSS snippets        : chips, cards and the global stylesheet.
"""

import base64
import hashlib
import html
import re
from pathlib import Path
from typing import Dict, List, Optional

# =============================================================================
# 1. PDF -> profile parsing
# =============================================================================

RESUME_SECTIONS = ["PROFESSIONAL SUMMARY", "TECHNICAL SKILLS", "SOFTWARE PROFICIENCY",
                   "PROFESSIONAL EXPERIENCE", "EDUCATION", "CERTIFICATIONS",
                   "ACHIEVEMENTS", "REFERENCES"]
SMALL_WORDS = {"of", "and", "the", "for", "to", "in"}


def read_pdf(path: Path) -> str:
    """Extract text from a PDF with pypdf (already in requirements.txt)."""
    from pypdf import PdfReader
    return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)


def _clean(line: str) -> str:
    """Strip bullet glyphs / leading symbols left over from PDF extraction."""
    return re.sub(r"^[^\w$(]+", "", line).strip()


def _split_sections(text: str, headers: List[str]) -> Dict[str, List[str]]:
    """Group the lines of `text` under the most recent known header."""
    sections: Dict[str, List[str]] = {"_top": []}
    current = "_top"
    glued = "|".join(re.escape(h) for h in headers)
    text = re.sub(rf"(?<=[a-z.)])(?=(?:{glued})(?:\s|$))", "\n", text)
    for raw in text.splitlines():
        line = raw.strip()
        if line.upper() in headers:
            current = line.upper()
            sections[current] = []
        elif line:
            sections[current].append(line)
    return sections


def parse_resume(text: str, path: Path) -> Dict:
    """Turn one resume's text into a dict of display-ready fields."""
    sec = _split_sections(text, RESUME_SECTIONS)
    top = sec["_top"]
    name = top[0].title() if top else path.stem.replace("_", " ").title()
    title = top[1] if len(top) > 1 else ""

    def find(label):
        m = re.search(label + r":\s*(.+)", text)
        return m.group(1).strip() if m else ""

    years = re.search(r"(\d+)\+?\s+years?", " ".join(sec.get("PROFESSIONAL SUMMARY", [])))
    years = int(years.group(1)) if years else 0

    # Experience entries look like "Title | Company | 2021 - 2023" followed by bullets.
    jobs: List[Dict] = []
    for line in sec.get("PROFESSIONAL EXPERIENCE", []):
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 3:
            jobs.append({"role": parts[0], "company": parts[1], "dates": parts[2], "bullets": []})
        elif jobs:
            jobs[-1]["bullets"].append(_clean(line))

    edu = sec.get("EDUCATION", [])
    education = {"degree": edu[0] if edu else "",
                 "school": edu[1].split("|")[0].strip() if len(edu) > 1 else "",
                 "year": edu[1].split("|")[1].strip() if len(edu) > 1 and "|" in edu[1] else "",
                 "gpa": next((e.replace("GPA:", "").strip() for e in edu if e.startswith("GPA")), "")}

    level = "Entry" if years <= 1 else "Mid" if years <= 4 else "Senior" if years <= 7 else "Leadership"
    return {
        "id": f"c_{path.stem}", "file": str(path), "filename": path.name, "text": text,
        "name": name, "title": title, "location": find("Location"), "email": find("Email"),
        "phone": find("Phone"), "linkedin": find("LinkedIn"), "years": years, "level": level,
        "summary": " ".join(sec.get("PROFESSIONAL SUMMARY", [])),
        "skills": [_clean(s) for s in sec.get("TECHNICAL SKILLS", [])],
        "software": [_clean(s) for s in sec.get("SOFTWARE PROFICIENCY", [])],
        "certs": [_clean(s) for s in sec.get("CERTIFICATIONS", [])],
        "jobs": jobs, "education": education,
    }


def _bullets(lines: List[str]) -> List[str]:
    """Bullet list from section lines. Handles both one-bullet-per-line PDFs and PDFs whose
    bullets run together in a paragraph ('• a • b • c')."""
    joined = " ".join(lines)
    if "\u2022" in joined:
        return [b.strip() for b in joined.split("\u2022") if b.strip()]
    return [_clean(x) for x in lines]


def parse_job(text: str, path: Path) -> Dict:
    """Extract header facts and skill lists from a job-description PDF."""
    flat = re.sub(r"\s+", " ", text)   # labels may wrap across lines ("Employment\nType:")

    def label(name):
        m = re.search(name + r":\s*(.+?)(?=\s+(?:Company|Location|Employment Type|Salary Range|Experience Required):|\s+JOB DESCRIPTION|$)", flat)
        return m.group(1).strip() if m else ""

    first = text.strip().splitlines()[0] if text.strip() else path.stem
    words = [w if "&" in w else (w.lower() if w.lower() in SMALL_WORDS else w.capitalize())
             for w in first.split()]
    title = " ".join(words)
    title = re.sub(r"\s+Position$", "", title)

    sec = _split_sections(text, ["JOB DESCRIPTION", "RESPONSIBILITIES", "REQUIRED SKILLS",
                                 "PREFERRED QUALIFICATIONS", "TECHNICAL SKILLS", "SOFT SKILLS", "BENEFITS"])
    exp = label("Experience Required")
    m = re.search(r"(\d+)", exp)
    return {"id": f"jd_{path.stem}", "file": str(path), "filename": path.name, "text": text,
            "title": title, "company": label("Company"), "location": label("Location"),
            "type": label("Employment Type"), "salary": label("Salary Range"),
            "experience": exp, "min_years": int(m.group(1)) if m else 0,
            "intro": " ".join(sec.get("JOB DESCRIPTION", [])),
            "responsibilities": _bullets(sec.get("RESPONSIBILITIES", [])),
            "required": _bullets(sec.get("REQUIRED SKILLS", [])),
            "preferred": _bullets(sec.get("PREFERRED QUALIFICATIONS", [])),
            "tech": _bullets(sec.get("TECHNICAL SKILLS", []))}


def load_profiles(directory: Path) -> List[Dict]:
    """Parse every resume PDF in `directory`, sorted by name."""
    out = []
    for p in sorted(Path(directory).glob("*.pdf")):
        try:
            out.append(parse_resume(read_pdf(p), p))
        except Exception:      # a corrupt PDF must not break the whole UI
            continue
    return sorted(out, key=lambda p: p["name"])


def load_jobs(directory: Path) -> List[Dict]:
    """Parse every job-description PDF in `directory`, sorted by title."""
    out = []
    for p in sorted(Path(directory).glob("*.pdf")):
        try:
            out.append(parse_job(read_pdf(p), p))
        except Exception:
            continue
    return sorted(out, key=lambda j: j["title"])


# =============================================================================
# 2. Pictures (generated SVG, embedded as data URIs)
# =============================================================================

PALETTES = [("#6366F1", "#8B5CF6"), ("#0EA5E9", "#6366F1"), ("#10B981", "#0EA5E9"),
            ("#F59E0B", "#EF4444"), ("#EC4899", "#8B5CF6"), ("#14B8A6", "#22C55E"),
            ("#F97316", "#EC4899"), ("#3B82F6", "#06B6D4")]


def _uri(svg: str) -> str:
    return "data:image/svg+xml;base64," + base64.b64encode(svg.encode()).decode()


def _palette(seed: str):
    return PALETTES[int(hashlib.md5(seed.encode()).hexdigest(), 16) % len(PALETTES)]


def avatar_uri(name: str) -> str:
    """Profile picture: coloured gradient, head-and-shoulders silhouette, initials.
    The colour is derived from the name, so a person always gets the same picture."""
    c1, c2 = _palette(name)
    ini = "".join(w[0] for w in name.split()[:2]).upper()
    return _uri(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient>
<clipPath id="c"><circle cx="48" cy="48" r="48"/></clipPath></defs>
<g clip-path="url(#c)"><rect width="96" height="96" fill="url(#g)"/>
<circle cx="48" cy="34" r="15" fill="#fff" fill-opacity=".30"/>
<path d="M12 96c2-24 17-38 36-38s34 14 36 38z" fill="#fff" fill-opacity=".30"/>
<text x="48" y="82" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="22" font-weight="700" fill="#fff">{ini}</text></g></svg>''')


def company_uri(company: str) -> str:
    """Square company logo placeholder built from the company's initials."""
    c1, c2 = _palette(company)
    ini = "".join(w[0] for w in re.findall(r"[A-Za-z]+", company)[:2]).upper() or "HF"
    return _uri(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 96 96"><defs>
<linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="{c1}"/><stop offset="1" stop-color="{c2}"/></linearGradient></defs>
<rect width="96" height="96" rx="20" fill="url(#g)"/><rect x="26" y="20" width="44" height="30" rx="4" fill="#fff" fill-opacity=".28"/>
<text x="48" y="78" text-anchor="middle" font-family="Arial,Helvetica,sans-serif" font-size="26" font-weight="700" fill="#fff">{ini}</text></svg>''')


def hero_uri() -> str:
    """Banner illustration: gradient with floating profile cards and a magnifier."""
    cards = "".join(
        f'<g transform="translate({x} {y}) rotate({r})"><rect width="150" height="86" rx="12" fill="#fff" fill-opacity=".16"/>'
        f'<circle cx="30" cy="32" r="16" fill="#fff" fill-opacity=".45"/><rect x="56" y="22" width="76" height="8" rx="4" fill="#fff" fill-opacity=".55"/>'
        f'<rect x="56" y="40" width="54" height="7" rx="3.5" fill="#fff" fill-opacity=".35"/><rect x="16" y="62" width="118" height="8" rx="4" fill="#fff" fill-opacity=".25"/></g>'
        for x, y, r in [(760, 26, -6), (930, 70, 5), (700, 128, 4)])
    return _uri(f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1200 240" preserveAspectRatio="xMaxYMid slice">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#312E81"/><stop offset=".55" stop-color="#4F46E5"/><stop offset="1" stop-color="#0EA5E9"/></linearGradient></defs>
<rect width="1200" height="240" fill="url(#g)"/><circle cx="1090" cy="40" r="120" fill="#fff" fill-opacity=".06"/><circle cx="620" cy="250" r="140" fill="#fff" fill-opacity=".05"/>
{cards}<g transform="translate(1030 120)" fill="none" stroke="#fff" stroke-opacity=".8" stroke-width="10" stroke-linecap="round"><circle r="34"/><path d="M26 26l34 34"/></g></svg>''')


# =============================================================================
# 3. Matching helpers
# =============================================================================

def _norm(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").lower()).strip()


class ProfileIndex:
    """Finds the profile behind a search result, however the engine labelled it.

    Vector results carry the real candidate_id; BM25 results carry a positional id
    ('c_3') and a name taken from the first text line. We therefore try, in order:
    candidate_id -> normalised name -> first 300 chars of the text."""

    def __init__(self, profiles: List[Dict]):
        self.by_id = {p["id"]: p for p in profiles}
        self.by_name = {_norm(p["name"]): p for p in profiles}
        self.by_text = {_norm(p["text"])[:300]: p for p in profiles}

    def resolve(self, result: Dict) -> Optional[Dict]:
        meta = result.get("metadata") or {}
        cid = result.get("candidate_id") or meta.get("candidate_id")
        name = result.get("name") or meta.get("name") or ""
        text = result.get("text") or result.get("page_content") or ""
        return (self.by_id.get(cid)
                or self.by_name.get(_norm(name))
                or self.by_name.get(_norm(re.sub(r"\s*resume\s*\d*$", "", name, flags=re.I)))
                or self.by_text.get(_norm(text)[:300]))


def skill_terms(p: Dict) -> List[str]:
    """Everything on a resume that can be matched against a job posting."""
    return p["skills"] + p["software"] + p["certs"]


def skill_match(p: Dict, jd_text: str) -> Dict:
    """Which of the candidate's skills/tools/certs does the job posting mention?"""
    jd = jd_text.lower().replace("microsoft ", "")
    hit, miss = [], []
    for t in skill_terms(p):
        (hit if t.lower().replace("microsoft ", "") in jd else miss).append(t)
    total = len(hit) + len(miss)
    return {"matched": hit, "other": miss, "pct": round(100 * len(hit) / total) if total else 0}


# =============================================================================
# 4. Offline search (BM25 only) - used when API keys / Pinecone are unavailable
# =============================================================================

STOP = set("a an and are as at be by for from has have in is it of on or the to with we our you your will "
           "this that their they can not but than more such using use who".split())


def tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9&+#]+", text.lower()) if t not in STOP and len(t) > 1]


class LocalSearch:
    """Keyword-only ranking over the parsed profiles (rank-bm25)."""

    def __init__(self, profiles: List[Dict]):
        from rank_bm25 import BM25Okapi
        self.profiles = profiles
        self.bm25 = BM25Okapi([tokens(p["text"]) for p in profiles])

    def search(self, query: str, k: int) -> List[Dict]:
        scores = self.bm25.get_scores(tokens(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [{"candidate_id": self.profiles[i]["id"], "score": float(scores[i])}
                for i in order if scores[i] > 0]


# =============================================================================
# 5. HTML / CSS snippets
# =============================================================================

def chips(items: List[str], kind: str = "skill", limit: int = 0, highlight=()) -> str:
    """Render tags. `highlight` items (matched skills) get the green style."""
    hl = {h.lower() for h in highlight}
    shown = items[:limit] if limit else items
    out = "".join(f'<span class="chip {"match" if i.lower() in hl else kind}">{html.escape(i)}</span>' for i in shown)
    if limit and len(items) > limit:
        out += f'<span class="chip more">+{len(items) - limit}</span>'
    return out


def bar(pct: float, label: str) -> str:
    pct = max(0, min(100, pct))
    return (f'<div class="bar-label"><span>{label}</span><b>{pct:.0f}%</b></div>'
            f'<div class="bar"><div style="width:{pct:.0f}%"></div></div>')


def blank_profile(result: Dict) -> Dict:
    """Minimal profile for a search result we could not match to a parsed PDF."""
    name = (result.get("name") or "Unknown Candidate").title()
    return {"id": result.get("candidate_id", "unknown"), "file": "", "filename": "", "text": result.get("text", ""),
            "name": name, "title": "", "location": "", "email": "", "phone": "", "linkedin": "", "years": 0,
            "level": "Entry", "summary": "", "skills": list(result.get("skills") or []), "software": [],
            "certs": [], "jobs": [], "education": {"degree": "", "school": "", "year": "", "gpa": ""}}


def meta_line(p: Dict) -> str:
    """'📍 City · 💼 6+ yrs · 🎓 Degree, School' (escaped)."""
    e = html.escape
    edu = p["education"]
    edu_txt = ", ".join(x for x in (edu.get("degree"), edu.get("school")) if x)
    parts = [f"📍 {e(p['location'] or 'Location n/a')}",
             f"💼 {p['years']}+ yrs" if p["years"] else "💼 Entry level"]
    if edu_txt:
        parts.append(f"🎓 {e(edu_txt)}")
    return " &nbsp;·&nbsp; ".join(parts)


def card_html(rank: int, p: Dict, highlight=()) -> str:
    """Result card header: rank, picture, name, title, facts and tag chips.
    Skills that appear in the job posting (`highlight`) are shown first, in green."""
    hl = {h.lower() for h in highlight}
    order = lambda items: sorted(items, key=lambda s: s.lower() not in hl)
    e = html.escape
    return (f'<div class="card"><div class="rank">#{rank}<small>match</small></div>'
            f'<img class="avatar" src="{avatar_uri(p["name"])}" alt="{e(p["name"])}">'
            f'<div class="body"><div class="name">{e(p["name"])}</div><div class="title">{e(p["title"])}</div>'
            f'<div class="meta">{meta_line(p)}</div>'
            f'{chips([p["level"] + " level"], "level")}{chips(order(p["skills"]), "skill", 8, hl)}'
            f'{chips(order(p["software"]), "sw", 5, hl)}{chips(p["certs"], "cert", 4, hl)}</div></div>')


def gallery_card_html(p: Dict) -> str:
    """Compact card for the talent-pool grid."""
    e = html.escape
    return (f'<div class="gallery"><img class="avatar" src="{avatar_uri(p["name"])}" alt="{e(p["name"])}">'
            f'<div class="name">{e(p["name"])}</div><div class="title">{e(p["title"])}</div>'
            f'<div class="meta">📍 {e(p["location"] or "n/a")} &nbsp;·&nbsp; 💼 {p["years"]}+ yrs</div>'
            f'{chips([p["level"]], "level")}{chips(p["skills"], "skill", 2)}</div>')


def timeline_html(jobs: List[Dict]) -> str:
    """Work history as a vertical timeline."""
    e = html.escape
    if not jobs:
        return "<i>No experience section found.</i>"
    items = ""
    for j in jobs:
        bullets = "".join(f"<li>{e(b)}</li>" for b in j["bullets"][:5])
        items += (f'<div class="job"><b>{e(j["role"])}</b> · {e(j["company"])}<br>'
                  f'<span>{e(j["dates"])}</span><ul>{bullets}</ul></div>')
    return f'<div class="tl">{items}</div>'


def avatar_strip(profiles: List[Dict], n: int = 12) -> str:
    """Row of overlapping profile pictures (hover shows the name)."""
    imgs = "".join(f'<img src="{avatar_uri(p["name"])}" title="{html.escape(p["name"])} - {html.escape(p["title"])}">'
                   for p in profiles[:n])
    return f'<div class="strip">{imgs}</div>'


def job_card_html(j: Dict) -> str:
    """Header card for a job description (company logo, facts, key tools)."""
    e = html.escape
    return (f'<div class="card"><img class="logo" src="{company_uri(j["company"])}" alt="{e(j["company"])}">'
            f'<div class="body"><div class="name">{e(j["title"])}</div><div class="title">{e(j["company"])}</div>'
            f'<div class="meta">📍 {e(j["location"])} &nbsp;·&nbsp; 💰 {e(j["salary"])} &nbsp;·&nbsp; 💼 {e(j["experience"])}</div>'
            f'{chips(j["tech"], "sw", 5)}</div></div>')


CSS = """
<style>
.block-container{padding-top:1.4rem;max-width:1200px}
.hf-hero{border-radius:18px;padding:34px 38px;color:#fff;margin-bottom:18px;background-size:cover;background-position:right center}
.hf-hero h1{margin:0 0 6px;font-size:2.1rem;color:#fff;line-height:1.15}
.hf-hero p{margin:0;max-width:560px;opacity:.92;font-size:1.02rem}
.hf-hero .pill{display:inline-block;margin:14px 8px 0 0;padding:4px 12px;border-radius:99px;background:rgba(255,255,255,.18);font-size:.8rem}
.card{display:flex;gap:16px;align-items:flex-start}
.avatar{width:72px;height:72px;border-radius:50%;flex:none;box-shadow:0 2px 8px rgba(0,0,0,.18)}
.logo{width:64px;height:64px;border-radius:14px;flex:none;box-shadow:0 2px 8px rgba(0,0,0,.18)}
.body{flex:1;min-width:0}
.name{font-size:1.15rem;font-weight:700;line-height:1.2}
.title{opacity:.75;font-size:.92rem;margin-bottom:2px}
.meta{font-size:.85rem;opacity:.72;margin:4px 0 8px}
.rank{flex:none;text-align:center;min-width:54px;font-weight:700;font-size:1.3rem;opacity:.85}
.rank small{display:block;font-size:.65rem;font-weight:500;opacity:.7;text-transform:uppercase;letter-spacing:.05em}
.chip{display:inline-block;padding:2px 10px;margin:2px 4px 2px 0;border-radius:99px;font-size:.76rem;border:1px solid rgba(128,128,128,.28);background:rgba(128,128,128,.10)}
.chip.sw{background:rgba(14,165,233,.14);border-color:rgba(14,165,233,.4)}
.chip.cert{background:rgba(245,158,11,.16);border-color:rgba(245,158,11,.5)}
.chip.match{background:rgba(16,185,129,.18);border-color:rgba(16,185,129,.55);font-weight:600}
.chip.level{background:rgba(99,102,241,.16);border-color:rgba(99,102,241,.5)}
.chip.more{opacity:.6}
.strip{display:flex;align-items:center;padding-left:10px;margin:4px 0 8px}
.strip img{width:44px;height:44px;border-radius:50%;margin-left:-10px;border:2px solid rgba(255,255,255,.9);box-shadow:0 1px 4px rgba(0,0,0,.2)}
.bar-label{display:flex;justify-content:space-between;font-size:.78rem;opacity:.8;margin-top:6px}
.bar{height:7px;border-radius:99px;background:rgba(128,128,128,.22);overflow:hidden}
.bar>div{height:100%;border-radius:99px;background:linear-gradient(90deg,#6366F1,#0EA5E9)}
.gallery{text-align:center}
.gallery .avatar{width:84px;height:84px;margin:0 auto 6px;display:block}
.gallery .name{font-size:1rem}
.tl{border-left:2px solid rgba(128,128,128,.35);margin-left:6px;padding-left:16px}
.tl .job{margin-bottom:12px}.tl .job b{font-size:.95rem}.tl .job span{opacity:.65;font-size:.82rem}
.tl ul{margin:4px 0 0 0;padding-left:18px;font-size:.85rem;opacity:.85}
</style>
"""
