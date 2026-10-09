"""Helpers for editing docs/thesis/thesis.docx in place with python-docx.

The docx is the source of truth for the thesis. Rules come from the official
Purdue Standard Thesis Handbook (August 2026):
  - work inside the template and never override its styles
  - body text: 'Indented Paragraph' (Times New Roman 12 pt, 1.5 spacing, justified)
  - chapter titles: 'Heading 1', ALL CAPS; subheadings 'Heading 2'..'Heading 5', sentence case
  - no empty paragraphs for spacing; a chapter ends with a page break placed inside
    its last paragraph
  - sub/superscripts are real character formatting, never Unicode sub/superscript glyphs

Text markup accepted by these helpers:  PM_{2.5}  R^{2}  m^{3}
"""
import re
from copy import deepcopy

from docx.oxml import OxmlElement
from docx.oxml.ns import qn

THESIS = "docs/thesis/thesis.docx"
W14 = "http://schemas.microsoft.com/office/word/2010/wordml"
_MARK = re.compile(r"([_^])\{([^}]*)\}")


def el_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def set_text(el, text):
    """Plain replacement: put `text` in the first text run and blank the others."""
    ts = list(el.iter(qn("w:t")))
    if not ts:
        raise ValueError("element has no text run")
    ts[0].text = text
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in ts[1:]:
        t.text = ""


def para_style(p_el):
    ps = p_el.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return ps.get(qn("w:val")) if ps is not None else None


def _run(text, vert=None):
    r = OxmlElement("w:r")
    if vert:
        rpr = OxmlElement("w:rPr")
        va = OxmlElement("w:vertAlign")
        va.set(qn("w:val"), vert)
        rpr.append(va)
        r.append(rpr)
    t = OxmlElement("w:t")
    t.text = text
    t.set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    r.append(t)
    return r


def set_rich_text(p_el, marked):
    """Replace a paragraph's runs with `marked` text; _{..} and ^{..} become real
    subscript/superscript runs. Paragraph properties are untouched."""
    for child in list(p_el):
        if child.tag != qn("w:pPr"):
            p_el.remove(child)
    pos = 0
    for m in _MARK.finditer(marked):
        if m.start() > pos:
            p_el.append(_run(marked[pos:m.start()]))
        p_el.append(_run(m.group(2), "subscript" if m.group(1) == "_" else "superscript"))
        pos = m.end()
    if pos < len(marked):
        p_el.append(_run(marked[pos:]))


def add_page_break(p_el):
    r = OxmlElement("w:r")
    br = OxmlElement("w:br")
    br.set(qn("w:type"), "page")
    r.append(br)
    p_el.append(r)


def new_para(proto, marked):
    """Copy a prototype paragraph's style, with fresh text and no inherited ids."""
    new = deepcopy(proto)
    for attr in ("{%s}paraId" % W14, "{%s}textId" % W14):
        if attr in new.attrib:
            del new.attrib[attr]
    set_rich_text(new, marked)   # also drops the prototype's bookmarks and runs
    return new


def find_block(body, startswith, after=0):
    for i, el in enumerate(list(body)):
        if i >= after and el_text(el).strip().startswith(startswith):
            return i, el
    raise KeyError(startswith)


def chapter_bounds(doc, heading_text):
    els = list(doc.element.body)
    start = next(i for i, e in enumerate(els)
                 if e.tag == qn("w:p") and para_style(e) == "Heading1"
                 and el_text(e).strip() == heading_text)
    end = next(i for i in range(start + 1, len(els))
               if els[i].tag == qn("w:p")
               and para_style(els[i]) in ("Heading1", "Appendixheading", "MajorHeading"))
    return els, start, end


def replace_chapter_body(doc, heading_text, blocks, protos):
    """Replace everything between the Heading 1 `heading_text` and the next heading.

    blocks: list of (kind, marked_text); kind is a key of `protos`.
    The chapter's closing page break is re-attached to the last new paragraph.
    """
    body = doc.element.body
    els, start, end = chapter_bounds(doc, heading_text)
    had_break = False
    for e in els[start + 1:end]:
        if e.find(".//" + qn("w:sectPr")) is not None:
            raise RuntimeError("section break inside chapter body; handle by hand")
        if any(b.get(qn("w:type")) == "page" for b in e.iter(qn("w:br"))):
            had_break = True
        body.remove(e)
    anchor = els[start]
    for kind, text in blocks:
        new = new_para(protos[kind], text)
        anchor.addnext(new)
        anchor = new
    if had_break:
        add_page_break(anchor)
    return anchor
