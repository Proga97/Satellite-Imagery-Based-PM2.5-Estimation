#!/usr/bin/env python
"""Format checks for docs/thesis/thesis.docx against the Purdue Standard Thesis Handbook.

Structural checks on the docx, then a LibreOffice render for page-level checks.
LibreOffice is a sanity check only; Word is the final arbiter.
"""
import collections
import re
import subprocess
import sys
import zipfile
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from lxml import etree

ROOT = Path(__file__).resolve().parents[1]
THESIS = ROOT / "docs/thesis/thesis.docx"
TEMPLATE = ROOT / "docs/thesis/Purdue-Standard-Thesis-Template-August-2026.docx"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
OUTDIR = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data/reference/render"
problems = []


def bad(msg):
    problems.append(msg); print("  PROBLEM:", msg)


def text(e):
    return "".join(t.text or "" for t in e.iter(qn("w:t")))


def style(e):
    ps = e.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return ps.get(qn("w:val")) if ps is not None else "Normal"


print("== package parts identical to the template")
zt, zw = zipfile.ZipFile(TEMPLATE), zipfile.ZipFile(THESIS)
c14 = lambda b: etree.tostring(etree.fromstring(b), method="c14n")
for n in ["word/styles.xml", "word/settings.xml", "word/numbering.xml", "word/fontTable.xml"] + \
        [x for x in zt.namelist() if re.match(r"word/(header|footer)\d+\.xml", x)]:
    a, b = zt.read(n), zw.read(n)
    if n == "word/settings.xml":   # the builder adds only the update-fields-on-open flag
        b = re.sub(rb'<w:updateFields w:val="true"/>', b"", b)
    if c14(a) != c14(b):
        bad(f"{n} differs from the template")
print("  styles, settings, numbering, fonts, headers, footers: checked")

d = Document(str(THESIS))
body = list(d.element.body)
paras = [e for e in d.element.body.iter(qn("w:p"))]

print("== leftovers and markup")
full = "\n".join(text(e) for e in paras if not style(e).startswith("TOC") and style(e) != "TableofFigures")   # TOC entries are cached field results, refreshed in Word
for needle in ["Paragraph starts here", "TYPE YOUR", "First level subheading", "Choose ", "to be written",
               "[@", "{fig:", "{tab:", "{eq:", "_{", "^{", "No table of figures"]:
    n = full.count(needle)
    if n and needle != "No table of figures":
        bad(f"{n} x leftover text {needle!r}")
if re.search(r"[₀-₉⁰-⁹]", full):
    bad("Unicode sub/superscript digits present")
print("  placeholders, unresolved markup, Unicode sub/superscripts: checked")

print("== paragraphs")
allowed = {"Normal", "IndentedParagraph", "Heading1", "Heading2", "Heading3", "MajorHeading", "CaptionFigure",
           "CaptionTable", "FigurePicture", "NotesTable", "NotesFigure", "References", "Equation", "TOC1", "TOC2",
           "TOC3", "TOC4", "TOC5", "TableofFigures"}
used = collections.Counter(style(e) for e in body if e.tag == qn("w:p") and text(e).strip())
for s in used:
    if s not in allowed:
        bad(f"unexpected paragraph style {s}")
print("  styles in use:", dict(used))
# empty top-level paragraphs must be page-break holders (as in the template) or title-page spacing
first_major = next(i for i, e in enumerate(body) if style(e) == "MajorHeading")
empties = [i for i, e in enumerate(body) if e.tag == qn("w:p") and i > first_major and not text(e).strip()
           and not any(b.get(qn("w:type")) == "page" for b in e.iter(qn("w:br")))
           and e.find(".//" + qn("w:drawing")) is None and e.find(".//" + qn("w:sectPr")) is None
           and not list(e.iter(qn("w:fldChar"))) and not list(e.iter(qn("w:fldSimple")))
           and e.find(".//{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath") is None]
if empties:
    bad(f"{len(empties)} empty paragraphs after the front matter (indices {empties[:8]})")
ids = [e.get("{%s}paraId" % W14) for e in paras if e.get("{%s}paraId" % W14)]
if len(ids) != len(set(ids)):
    bad("duplicate paragraph ids")
names = [b.get(qn("w:name")) for b in d.element.body.iter(qn("w:bookmarkStart"))]
if len(names) != len(set(names)):
    bad("duplicate bookmark names")
for e in body:
    if e.tag == qn("w:p") and style(e) in ("IndentedParagraph", "Heading1", "Heading2", "Heading3"):
        for r in e.iter(qn("w:r")):
            rpr = r.find(qn("w:rPr"))
            if rpr is not None and (rpr.find(qn("w:rFonts")) is not None or rpr.find(qn("w:sz")) is not None):
                bad(f"font/size override in body text: {text(e)[:50]!r}"); break
print("  empty paragraphs, ids, bookmarks, font overrides: checked")

print("== headings")
h1 = [text(e) for e in body if style(e) == "Heading1"]
for t in h1:
    if t != t.upper():
        bad(f"chapter title not in capitals: {t}")
SMALL_OK = {"PM2.5", "AOD-driven", "SDG", "Sentinel-2", "ERA5-Land", "FiLM", "RQ1", "RQ2", "EPA", "ResNet-18"}
for e in body:
    if style(e) in ("Heading2", "Heading3"):
        words = text(e).split()
        caps = [w for w in words[1:] if w[0].isupper() and w.strip(",.()") not in SMALL_OK and not w.isupper()]
        if caps:
            bad(f"subheading may not be sentence case: {text(e)!r} ({caps})")
levels = [int(style(e)[-1]) for e in body if style(e) in ("Heading1", "Heading2", "Heading3")]
for a, b in zip(levels, levels[1:]):
    if b > a + 1:
        bad("heading level skipped")
print(f"  {len(h1)} chapters: {h1}")

print("== figures, tables, captions")
figs = [e for e in body if style(e) == "FigurePicture"]
caps_f = [e for e in body if style(e) == "CaptionFigure"]
caps_t = [e for e in body if style(e) == "CaptionTable"]
tables = [e for i, e in enumerate(body) if e.tag == qn("w:tbl") and i > first_major and style(body[i - 1]) == "CaptionTable"]
for i, e in enumerate(body):
    if style(e) == "FigurePicture":
        if style(body[i + 1]) != "CaptionFigure":
            bad("figure not followed by its caption")
        descr = [x.get("descr") for x in e.iter() if x.tag.endswith("}docPr")]
        if not descr or not descr[0] or len(descr[0]) < 30:
            bad("figure without alt text")
        ext = next(x for x in e.iter() if x.tag.endswith("}extent"))
        if int(ext.get("cx")) / 914400 > 6.5:
            bad("figure wider than the 6.5 in text block")
    if style(e) == "CaptionTable" and body[i + 1].tag != qn("w:tbl"):
        bad("table caption not followed by a table")
    if style(e) in ("CaptionFigure", "CaptionTable"):
        instr = [f.get(qn("w:instr")).strip() for f in e.iter(qn("w:fldSimple"))]
        if len(instr) != 2 or not instr[0].startswith("STYLEREF 1") or not instr[1].startswith("SEQ"):
            bad(f"caption is not a Word caption field: {text(e)[:40]!r}")
pic_ids = [x.get("id") for e in figs for x in e.iter() if x.tag.endswith("}docPr")]
if len(pic_ids) != len(set(pic_ids)):
    bad("duplicate picture ids")
for t in tables:
    if t.find(qn("w:tr")).find(qn("w:trPr")).find(qn("w:tblHeader")) is None:
        bad("table without a marked header row")
    for sz in t.iter(qn("w:sz")):
        if int(sz.get(qn("w:val"))) not in (16, 18, 20):
            bad("table font size outside 8-10 pt"); break
print(f"  {len(figs)} figures, {len(caps_f)} figure captions, {len(tables)} tables, {len(caps_t)} table captions")

print("== references")
refs = [e for e in body if style(e) == "References"]
nolink = [text(e)[:50] for e in refs if e.find(qn("w:hyperlink")) is None and text(e).strip()]
if nolink:
    bad(f"{len(nolink)} references without a hyperlink")
cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", full)})
i_ref = next(i for i, x in enumerate(body) if style(x) == "MajorHeading" and text(x).strip() == "REFERENCES")
i_end = next((i for i in range(i_ref + 1, len(body)) if style(body[i]) == "MajorHeading"), len(body))
nref = sum(1 for e in body[i_ref:i_end] if style(e) == "References" and text(e).strip())
if cited and (cited[0] != 1 or cited[-1] != nref or len(cited) != nref):
    bad(f"citation numbers {cited[0]}..{cited[-1]} ({len(cited)} distinct) do not match {nref} references")
print(f"  {nref} references, citations [1]..[{cited[-1] if cited else 0}]")

print("== render (LibreOffice)")
OUTDIR.mkdir(parents=True, exist_ok=True)
subprocess.run(["/Applications/LibreOffice.app/Contents/MacOS/soffice", "--headless", "--convert-to", "pdf",
                "--outdir", str(OUTDIR), str(THESIS)], capture_output=True, timeout=600)
import pymupdf
pdf = pymupdf.open(str(OUTDIR / "thesis.pdf"))
majors = [text(e).strip() for e in body if style(e) == "MajorHeading"]
starts = {}
fonts = collections.Counter()
for i in range(len(pdf)):
    lines = [l.strip() for l in pdf[i].get_text().split("\n") if l.strip() and not l.strip().isdigit()]
    for m in majors:
        for k, l in enumerate(lines):
            if l == m and k != 0 and m not in starts:
                bad(f"{m} does not start its page (page {i + 1})")
            if l == m and k == 0:
                starts[m] = i + 1
    for k, l in enumerate(lines):
        mm = re.match(r"^(\d)\. ([A-Z][A-Z ,]+)$", l)
        if mm and mm.group(2).strip() in h1:
            starts[l] = i + 1
            if k != 0:
                bad(f"chapter {l} does not start its page (page {i + 1})")
    for b in pdf[i].get_text("dict")["blocks"]:
        for ln in b.get("lines", []):
            for s in ln["spans"]:
                if s["text"].strip():
                    fonts[s["font"]] += len(s["text"])
print(f"  pages: {len(pdf)}")
print("  section starts:", {k: v for k, v in sorted(starts.items(), key=lambda kv: kv[1])})
print("  fonts (characters):", dict(fonts.most_common()))
other = {f: n for f, n in fonts.items() if not f.startswith(("TimesNewRoman", "CambriaMath", "Cambria Math"))}
if other:
    print("  NOTE non-Times fonts in render (may be LibreOffice substitution):", other)
print()
print("RESULT:", "no problems found" if not problems else f"{len(problems)} problem(s)")
