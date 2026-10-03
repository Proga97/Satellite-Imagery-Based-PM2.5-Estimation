"""Helpers for editing docs/thesis/thesis.docx in place with python-docx.

The docx is the source of truth for the thesis. These helpers keep the Purdue
template's styles: body text is 'Indented Paragraph', chapter titles 'Heading 1',
sections 'Heading 2'/'Heading 3', captions 'Caption (Figure)'/'Caption (Table)'.
"""
from copy import deepcopy

from docx import Document
from docx.oxml.ns import qn

THESIS = "docs/thesis/thesis.docx"


def el_text(el):
    return "".join(t.text or "" for t in el.iter(qn("w:t")))


def set_text(el, text):
    """Put `text` in the first text run of an element and blank the others."""
    ts = list(el.iter(qn("w:t")))
    if not ts:
        raise ValueError("element has no text run")
    ts[0].text = text
    ts[0].set("{http://www.w3.org/XML/1998/namespace}space", "preserve")
    for t in ts[1:]:
        t.text = ""


def find_block(body, startswith, tag=None, after=0):
    for i, el in enumerate(list(body)):
        if i < after:
            continue
        if tag and el.tag.split("}")[1] != tag:
            continue
        if el_text(el).strip().startswith(startswith):
            return i, el
    raise KeyError(startswith)


def para_style(p_el):
    ps = p_el.find(qn("w:pPr") + "/" + qn("w:pStyle"))
    return ps.get(qn("w:val")) if ps is not None else None


def clone_with_text(proto, text):
    new = deepcopy(proto)
    runs = new.findall(qn("w:r"))
    for r in runs[1:]:
        new.remove(r)
    set_text(new, text)
    return new


def replace_chapter_body(doc, heading_text, blocks, protos):
    """Replace everything between the Heading 1 `heading_text` and the next Heading 1.

    blocks: list of (kind, text) with kind in protos (e.g. 'body', 'h2', 'h3').
    protos: dict kind -> prototype paragraph element (deep-copied per block).
    """
    body = doc.element.body
    els = list(body)
    start = next(i for i, e in enumerate(els)
                 if e.tag == qn("w:p") and para_style(e) == "Heading1"
                 and el_text(e).strip() == heading_text)
    end = next(i for i in range(start + 1, len(els))
               if els[i].tag == qn("w:p") and para_style(els[i]) in ("Heading1", "Appendixheading", "MajorHeading"))
    for e in els[start + 1:end]:
        if e.tag != qn("w:p"):
            continue
        if e.find(".//" + qn("w:sectPr")) is None and e.find(".//" + qn("w:br")) is None:
            body.remove(e)
        else:  # carries a page/section break: keep the break, drop placeholder text
            for t in e.iter(qn("w:t")):
                t.text = ""
    anchor = els[start]
    for kind, text in blocks:
        new = clone_with_text(protos[kind], text)
        anchor.addnext(new)
        anchor = new
    return start
