"""Small reportlab helpers used by build_report.py (layout only, no numbers)."""
from __future__ import annotations

import os

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

PAGE_W, PAGE_H = A4
MARGIN = 1.8 * cm
TEXT_W = PAGE_W - 2 * MARGIN

_ss = getSampleStyleSheet()
S = {
    "title": ParagraphStyle("t", parent=_ss["Title"], fontSize=22, leading=27, spaceAfter=10),
    "h1": ParagraphStyle("h1", parent=_ss["Heading1"], fontSize=15, leading=19, spaceBefore=10, spaceAfter=6,
                         textColor=colors.HexColor("#14325c")),
    "h2": ParagraphStyle("h2", parent=_ss["Heading2"], fontSize=12.5, leading=16, spaceBefore=8, spaceAfter=4,
                         textColor=colors.HexColor("#14325c")),
    "h3": ParagraphStyle("h3", parent=_ss["Heading3"], fontSize=10.5, leading=13.5, spaceBefore=6, spaceAfter=2),
    "body": ParagraphStyle("b", parent=_ss["BodyText"], fontSize=9.3, leading=12.6, spaceAfter=4),
    "small": ParagraphStyle("s", parent=_ss["BodyText"], fontSize=7.6, leading=9.6),
    "cell": ParagraphStyle("c", parent=_ss["BodyText"], fontSize=7.4, leading=9.0),
    "cellb": ParagraphStyle("cb", parent=_ss["BodyText"], fontSize=7.4, leading=9.0, fontName="Helvetica-Bold"),
    "cap": ParagraphStyle("cap", parent=_ss["BodyText"], fontSize=8, leading=10, textColor=colors.HexColor("#444444"),
                          alignment=TA_CENTER, spaceAfter=8),
    "center": ParagraphStyle("ctr", parent=_ss["BodyText"], fontSize=11, leading=15, alignment=TA_CENTER),
    "bullet": ParagraphStyle("bl", parent=_ss["BodyText"], fontSize=9.3, leading=12.4, leftIndent=12,
                             bulletIndent=2, spaceAfter=1.5),
}


def P(text, style="body"):
    return Paragraph(text, S[style])


def bullets(items):
    return [Paragraph(t, S["bullet"], bulletText="•") for t in items]


def img(path, width=TEXT_W, caption=None, max_h=None):
    w, h = PILImage.open(path).size
    width = min(width, TEXT_W)
    height = width * h / w
    if max_h and height > max_h:
        height = max_h
        width = height * w / h
    parts = [Image(path, width=width, height=height)]
    if caption:
        parts.append(Paragraph(caption, S["cap"]))
    return KeepTogether(parts)


def table(rows, col_widths=None, header_rows=1, font_size=7.4, zebra=True, bold_first_col=False):
    data = []
    for i, r in enumerate(rows):
        st = "cellb" if i < header_rows else "cell"
        data.append([c if not isinstance(c, str) else Paragraph(c, S["cellb"] if (bold_first_col and j == 0) else S[st])
                     for j, c in enumerate(r)])
    t = Table(data, colWidths=col_widths, repeatRows=header_rows)
    style = [("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#b8c2cc")),
             ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dde7f3")),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
             ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2)]
    if zebra:
        for i in range(header_rows, len(rows)):
            if (i - header_rows) % 2:
                style.append(("BACKGROUND", (0, i), (-1, i), colors.HexColor("#f5f8fb")))
    t.setStyle(TableStyle(style))
    return t


def build(path, story, title):
    def on_page(c, doc):
        c.saveState()
        c.setFont("Helvetica", 7.5)
        c.setFillColor(colors.HexColor("#666666"))
        if doc.page > 1:
            c.drawString(MARGIN, 1.0 * cm, title)
            c.drawRightString(PAGE_W - MARGIN, 1.0 * cm, f"page {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN, topMargin=1.6 * cm,
                            bottomMargin=1.6 * cm, title=title, author="Team XX")
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)


__all__ = ["P", "bullets", "img", "table", "build", "S", "TEXT_W", "cm", "Spacer", "PageBreak", "KeepTogether"]
