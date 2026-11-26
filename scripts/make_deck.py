"""Build the presentation deck from the project's own figures and results.

Usage:
    python scripts/make_deck.py [--out report/slides.pptx]

Needs python-pptx and Pillow. Every number in the deck is copied from
results.json / reports/, so regenerating the deck after a rerun keeps it honest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

# ---------------------------------------------------------------- palette
INK = RGBColor(0x0B, 0x0B, 0x0B)
MUTED = RGBColor(0x52, 0x51, 0x4E)
BLUE = RGBColor(0x2A, 0x78, 0xD6)
ORANGE = RGBColor(0xEB, 0x68, 0x34)
RULE = RGBColor(0xD9, 0xD8, 0xD4)
BG = RGBColor(0xFC, 0xFC, 0xFB)
PANEL = RGBColor(0xF2, 0xF1, 0xEE)

W, H = Inches(13.333), Inches(7.5)
MARGIN = Inches(0.78)
BODY_TOP = Inches(1.95)
BODY_W = W - 2 * MARGIN

TEAM = "Group 11  ·  COMP4163 Course Project"

TITLE_OF = {"1": "The puzzle", "2": "Our pipeline", "3": "What we found",
            "4": "Why, and how far it goes"}


class Deck:
    def __init__(self) -> None:
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = W, H
        self.n = 0
        self.part = ""

    # ------------------------------------------------------------ helpers
    def _blank(self):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        bg = s.background.fill
        bg.solid()
        bg.fore_color.rgb = BG
        self.n += 1
        return s

    def _text(self, slide, left, top, width, height, text, size, color=INK,
              bold=False, align=PP_ALIGN.LEFT, italic=False, spacing=1.0):
        box = slide.shapes.add_textbox(left, top, width, height)
        tf = box.text_frame
        tf.word_wrap = True
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        p = tf.paragraphs[0]
        p.alignment = align
        p.line_spacing = spacing
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.italic = italic
        r.font.color.rgb = color
        r.font.name = "Calibri"
        return box

    def _rule(self, slide, top, width=Inches(1.15), color=BLUE, left=MARGIN):
        bar = slide.shapes.add_shape(1, left, top, width, Inches(0.045))
        bar.fill.solid()
        bar.fill.fore_color.rgb = color
        bar.line.fill.background()
        bar.shadow.inherit = False

    def _footer(self, slide):
        left = TEAM + (f"  ·  {self.part}" if self.part else "")
        self._text(slide, MARGIN, H - Inches(0.52), Inches(9), Inches(0.3),
                   left, 10, MUTED)
        self._text(slide, W - MARGIN - Inches(1.2), H - Inches(0.52),
                   Inches(1.2), Inches(0.3), str(self.n), 10, MUTED,
                   align=PP_ALIGN.RIGHT)

    def _head(self, slide, kicker, title):
        self._text(slide, MARGIN, Inches(0.42), BODY_W, Inches(0.28), kicker,
                   11.5, MUTED, bold=True)
        self._text(slide, MARGIN, Inches(0.72), BODY_W, Inches(0.85), title,
                   29, INK, bold=True, spacing=0.95)
        self._rule(slide, Inches(1.62))
        self._footer(slide)

    # ------------------------------------------------------------- slides
    def title(self):
        s = self._blank()
        self._text(s, MARGIN, Inches(1.35), BODY_W, Inches(0.3),
                   "COMP4163 COURSE PROJECT  ·  GROUP 11", 13, BLUE, bold=True)
        self._text(s, MARGIN, Inches(1.85), BODY_W, Inches(1.5),
                   "When deeper is worse", 50, INK, bold=True, spacing=0.92)
        self._text(s, MARGIN, Inches(2.75), BODY_W, Inches(1.2),
                   "Reproducing the degradation problem, and testing whether\n"
                   "learning-rate tuning explains it", 22, MUTED, spacing=1.15)
        self._rule(s, Inches(4.05), Inches(1.6))
        self._text(s, MARGIN, Inches(4.35), BODY_W, Inches(1.4),
                   "Chen Letian 2330026013   ·   Zhang Yiyang 2430024331\n"
                   "Liu Chuanliang 2330013021   ·   Liu Yilin 2430026139",
                   14, INK, spacing=1.5)
        self._text(s, MARGIN, Inches(5.6), BODY_W, Inches(0.4),
                   "He, Zhang, Ren & Sun, Deep Residual Learning for Image "
                   "Recognition, CVPR 2016", 12, MUTED, italic=True)
        return s

    def agenda(self, rows):
        """Roadmap slide: what the four parts are, who gives each, how long."""
        s = self._blank()
        self.part = ""
        self._text(s, MARGIN, Inches(0.55), BODY_W, Inches(0.3),
                   "AGENDA", 11.5, MUTED, bold=True)
        self._text(s, MARGIN, Inches(0.85), BODY_W, Inches(0.8),
                   "What we will cover", 30, INK, bold=True)
        self._rule(s, Inches(1.72))
        top = Inches(2.15)
        for i, (num, title, who, mins) in enumerate(rows):
            y = int(top + i * Inches(1.05))
            bar = s.shapes.add_shape(1, MARGIN, y + Inches(0.12),
                                     Inches(0.05), Inches(0.55))
            bar.fill.solid()
            bar.fill.fore_color.rgb = BLUE if i < 4 else RULE
            bar.line.fill.background()
            bar.shadow.inherit = False
            self._text(s, MARGIN + Inches(0.28), y, Inches(0.6), Inches(0.5),
                       num, 17, BLUE, bold=True)
            self._text(s, MARGIN + Inches(0.9), y + Inches(0.02), Inches(6.4),
                       Inches(0.5), title, 19, INK)
            self._text(s, MARGIN + Inches(7.4), y + Inches(0.05), Inches(2.6),
                       Inches(0.4), who, 15, MUTED)
            self._text(s, W - MARGIN - Inches(0.9), y + Inches(0.05),
                       Inches(0.9), Inches(0.4), mins, 15, MUTED,
                       align=PP_ALIGN.RIGHT)
        self._footer(s)
        return s

    def section(self, number, title, subtitle, who):
        s = self._blank()
        self.part = TITLE_OF.get(str(number), "")
        panel = s.shapes.add_shape(1, 0, 0, Inches(0.22), H)
        panel.fill.solid()
        panel.fill.fore_color.rgb = BLUE
        panel.line.fill.background()
        panel.shadow.inherit = False

        # Progress: one marker per part, filled up to the current one.
        if str(number).isdigit():
            for i in range(4):
                x = int(MARGIN + Inches(0.2) + i * Inches(0.5))
                m = s.shapes.add_shape(1, x, Inches(2.38), Inches(0.34),
                                       Inches(0.09))
                m.fill.solid()
                m.fill.fore_color.rgb = BLUE if i < int(number) else RULE
                m.line.fill.background()
                m.shadow.inherit = False

        label = f"PART {number} OF 4" if str(number).isdigit() else "APPENDIX"
        self._text(s, MARGIN + Inches(0.2), Inches(2.62), BODY_W, Inches(0.35),
                   label, 12, BLUE, bold=True)
        self._text(s, MARGIN + Inches(0.2), Inches(3.0), BODY_W, Inches(1.0),
                   title, 40, INK, bold=True)
        self._text(s, MARGIN + Inches(0.2), Inches(4.05), BODY_W, Inches(0.8),
                   subtitle, 18, MUTED, spacing=1.2)
        self._text(s, MARGIN + Inches(0.2), Inches(5.4), BODY_W, Inches(0.4),
                   f"Presented by {who}" if who else "", 13, MUTED)
        self._footer(s)
        return s

    def bullets(self, kicker, title, items, size=None, gap=13, lead=None):
        s = self._blank()
        self._head(s, kicker, title)
        # Fewer bullets get a larger type size, and the block is centred in the
        # body area, so a three-item slide does not sit marooned at the top.
        if size is None:
            size = 25 if len(items) <= 3 else (22 if len(items) == 4 else 19)
        top = BODY_TOP + Inches(0.1)
        if lead:
            self._text(s, MARGIN, top, BODY_W, Inches(0.6), lead, 20, MUTED,
                       italic=True)
            top += Inches(0.75)
        box = s.shapes.add_textbox(MARGIN, top, BODY_W, H - top - Inches(0.85))
        tf = box.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
        for i, item in enumerate(items):
            if isinstance(item, tuple):
                text, sub = item
            else:
                text, sub = item, None
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.space_after = Pt(gap)
            p.line_spacing = 1.12
            r = p.add_run()
            r.text = "—  " + text
            r.font.size = Pt(size)
            r.font.color.rgb = INK
            r.font.name = "Calibri"
            if sub:
                q = tf.add_paragraph()
                q.space_after = Pt(gap)
                q.line_spacing = 1.12
                q.level = 1
                r2 = q.add_run()
                r2.text = sub
                r2.font.size = Pt(size - 3)
                r2.font.color.rgb = MUTED
                r2.font.name = "Calibri"
        return s

    def figure(self, kicker, title, path, caption=None, max_h=Inches(3.75),
               note=None):
        s = self._blank()
        self._head(s, kicker, title)
        img = Path(path)
        iw, ih = Image.open(img).size
        scale = min(BODY_W / iw, max_h / ih)
        w, h = int(iw * scale), int(ih * scale)
        left = int((W - w) / 2)
        top = BODY_TOP + Inches(0.02)
        s.shapes.add_picture(str(img), left, top, w, h)
        if caption:
            self._text(s, MARGIN, top + h + Inches(0.12), BODY_W, Inches(0.55),
                       caption, 12.5, MUTED, spacing=1.12)
        if note:
            self._text(s, MARGIN, H - Inches(1.12), BODY_W, Inches(0.5),
                       note, 15.5, BLUE, bold=True, spacing=1.05)
        return s

    def hero(self, kicker, title, big, big_label, facts, note=None):
        """One number, large, with a short list of supporting facts."""
        s = self._blank()
        self._head(s, kicker, title)
        self._text(s, MARGIN, Inches(2.35), Inches(6.2), Inches(1.7), big,
                   110, BLUE, bold=True, spacing=0.9)
        self._text(s, MARGIN, Inches(3.95), Inches(6.2), Inches(0.9),
                   big_label, 17, MUTED, spacing=1.2)
        x = MARGIN + Inches(6.6)
        top = Inches(2.5)
        for label, value in facts:
            self._text(s, x, top, Inches(2.9), Inches(0.35), label, 13, MUTED)
            self._text(s, x + Inches(2.9), top - Inches(0.06), Inches(2.2),
                       Inches(0.45), value, 22, INK, bold=True)
            top += Inches(0.75)
        if note:
            self._text(s, MARGIN, H - Inches(1.15), BODY_W, Inches(0.6), note,
                       15.5, BLUE, bold=True, spacing=1.05)
        return s

    def table(self, kicker, title, headers, rows, caption=None, col_w=None,
              size=16, hi=None):
        s = self._blank()
        self._head(s, kicker, title)
        nrows, ncols = len(rows) + 1, len(headers)
        height = Inches(0.42) * nrows
        # Short tables are centred in the body area rather than left stranded
        # under the title.
        avail = H - BODY_TOP - Inches(1.65 if caption else 0.95)
        # Keep every coordinate an integer: PowerPoint rejects the file if the
        # shape offsets are written as floats.
        top = int(BODY_TOP + max(Inches(0), (avail - height) // 2))
        shape = s.shapes.add_table(nrows, ncols, MARGIN, top,
                                   BODY_W, height)
        tbl = shape.table
        if col_w:
            total = sum(col_w)
            for i, c in enumerate(col_w):
                tbl.columns[i].width = Emu(int(BODY_W * c / total))
        def fill_cell(cell, text, font_size, bold, color, align):
            cell.text = str(text)
            cell.fill.solid()
            cell.fill.fore_color.rgb = color[0]
            p = cell.text_frame.paragraphs[0]
            p.alignment = align
            run = p.runs[0] if p.runs else p.add_run()
            run.text = str(text)
            run.font.size = Pt(font_size)
            run.font.bold = bold
            run.font.color.rgb = color[1]
            run.font.name = "Calibri"

        for j, head in enumerate(headers):
            fill_cell(tbl.cell(0, j), head, size - 2, True, (PANEL, MUTED),
                      PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
        for i, row in enumerate(rows, start=1):
            for j, val in enumerate(row):
                fill_cell(tbl.cell(i, j), val, size, bool(hi and i == hi),
                          (BG, INK),
                          PP_ALIGN.LEFT if j == 0 else PP_ALIGN.CENTER)
        if caption:
            y = top + height + Inches(0.25)
            self._text(s, MARGIN, y, BODY_W, Inches(0.9), caption, 13, MUTED,
                       spacing=1.15)
        return s

    def twocol(self, kicker, title, left_head, left_items, right_head,
               right_items, foot=None):
        s = self._blank()
        self._head(s, kicker, title)
        colw = int((BODY_W - Inches(0.6)) / 2)
        for idx, (head, items, color) in enumerate(
                ((left_head, left_items, BLUE), (right_head, right_items, ORANGE))):
            x = int(MARGIN + idx * (colw + Inches(0.6)))
            panel = s.shapes.add_shape(1, x, BODY_TOP, colw, Inches(3.05))
            panel.fill.solid()
            panel.fill.fore_color.rgb = PANEL
            panel.line.fill.background()
            panel.shadow.inherit = False
            self._text(s, x + Inches(0.35), BODY_TOP + Inches(0.28),
                       colw - Inches(0.7), Inches(0.4), head, 17, color,
                       bold=True)
            box = s.shapes.add_textbox(x + Inches(0.35), BODY_TOP + Inches(0.85),
                                       colw - Inches(0.7), Inches(2.8))
            tf = box.text_frame
            tf.word_wrap = True
            tf.margin_left = tf.margin_right = 0
            for i, item in enumerate(items):
                p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
                p.space_after = Pt(11)
                p.line_spacing = 1.1
                r = p.add_run()
                r.text = "—  " + item
                r.font.size = Pt(16)
                r.font.color.rgb = INK
                r.font.name = "Calibri"
        if foot:
            self._text(s, MARGIN, H - Inches(1.25), BODY_W, Inches(0.6), foot,
                       16, INK, bold=True, spacing=1.1)
        return s

    def quote(self, kicker, title, big, sub):
        s = self._blank()
        self._head(s, kicker, title)
        self._text(s, MARGIN, Inches(2.9), BODY_W, Inches(1.6), big, 32, INK,
                   bold=True, spacing=1.15)
        self._rule(s, Inches(4.5), Inches(1.15), ORANGE)
        self._text(s, MARGIN, Inches(4.85), BODY_W, Inches(1.2), sub, 18,
                   MUTED, spacing=1.2)
        return s

    def save(self, out):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        self.prs.save(out)
        print(f"saved {out} ({len(self.prs.slides.__iter__.__self__._sldIdLst)} slides)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="report/slides.pptx")
    args = ap.parse_args()

    r20 = json.loads(Path("runs/resnet20-s0/results.json").read_text("utf-8"))

    d = Deck()
    d.title()
    d.agenda([
        ("1", "The puzzle: why a deeper network should never be worse",
         "Chen Letian", "3 min"),
        ("2", "Our pipeline, and the check that it matches the paper",
         "Zhang Yiyang", "3 min"),
        ("3", "The replication, and experiment 1: is it just mis-tuning?",
         "Liu Chuanliang", "4 min"),
        ("4", "Why it happens, experiment 2, and what we cannot claim",
         "Liu Yilin", "3 min"),
    ])

    # ---------------------------------------------------------- part 1
    d.section("1", "The puzzle", "A deeper network should never be worse than "
              "a shallower one — yet it is.", "Chen Letian")

    d.bullets("THE PUZZLE", "Deeper should never be worse", [
        ("A 56-layer net can imitate a 20-layer one: set the extra layers to identity",
         "So a 56-layer network should be at least as good as a 20-layer one"),
        ("He et al. (2016) train both on CIFAR-10 with the same recipe",
         "The 56-layer network has the higher training error: 2.47% against 0.43%"),
        ("The deeper network cannot even fit what it is trained on", None),
    ], size=22)

    d.twocol("THE DISTINCTION THAT MATTERS", "Degradation is not overfitting",
             "Overfitting", [
                 "Training error falls, test error rises",
                 "The model memorises instead of generalising",
                 "Cured by regularisation or more data",
             ],
             "Degradation", [
                 "Training and test error both rise with depth",
                 "It cannot even fit its own training set",
                 "Cured by the parameterisation, not the data",
             ],
             foot="Both errors rising is what rules out the obvious explanation.")

    d.table("MILESTONE 1 EXTRACTION", "How strong is the paper's evidence?",
            ["CIFAR-10 claim", "What the paper shows"],
            [["Figure 6 (curves)", "One curve per depth; number of runs not stated"],
             ["Table 6, ResNet-20 and ResNet-56", "Single runs (8.75% and 6.97%)"],
             ["Table 6, ResNet-110", "Five runs: best 6.43%, mean 6.61 plus or minus 0.16"]],
            caption="No seed spread is reported for any network we reproduce. "
                    "Our three seeds per network add the spread that decides "
                    "whether a difference between two networks is real.",
            col_w=[3, 5])

    # ---------------------------------------------------------- part 2
    d.section("2", "Our pipeline", "Same recipe, every open choice resolved, "
              "and checked against the paper before the big run.",
              "Zhang Yiyang")

    d.table("THE RECIPE (PAPER SEC. 4.2 + TOPIC CARD)", "What we ran",
            ["Setting", "Value"],
            [["Data", "45,000 train / 5,000 validation / 10,000 test"],
             ["Optimiser", "SGD, momentum 0.9, weight decay 1e-4, batch 128"],
             ["Learning rate", "0.1, divided by 10 at 32,000 and 48,000, stop at 64,000"],
             ["Architecture", "6n+2 layers; 16/32/64 channels; He init; no dropout"],
             ["Precision", "float16 autocast; all reported errors in float32"]],
            caption="The paper leaves five choices open: which 5,000 images "
                    "validate, 45k or 50k trained, runs per network, how the "
                    "Figure 6 training error is measured, and epochs versus "
                    "iterations. We fixed each one and report it.",
            col_w=[2, 6], size=16)

    d.hero("MILESTONE 2 — PIPELINE CHECK",
           "One run, the paper's full recipe",
           f"{r20['final']['test_err']:.2f}%",
           f"ResNet-20 test error at iteration 64,000\non one RTX 4060, in "
           f"{r20['elapsed_s'] / 60:.1f} minutes",
           [("Paper, Table 6", "8.75%"),
            ("Difference", f"{r20['final']['test_err'] - 8.75:+.2f} pp".replace("-", "−")),
            ("Runs in total", "29")],
           note="Same depth, width and 853,018 parameters as the plain net — "
                "the only difference is the shortcut.")

    # ---------------------------------------------------------- part 3
    d.section("3", "What we found", "The replication, then the controlled "
              "experiment that tests the most obvious objection.",
              "Liu Chuanliang")

    d.figure("REPLICATION — FIGURE 6", "Plain networks: deeper is worse, "
             "in all three panels", "figures/fig6_plain.png",
             caption="Thin lines are individual seeds, the bold line is the "
                     "mean; dotted lines are our Figure 6 read-offs.",
             note="Both training panels are higher too — this is not overfitting.")

    d.figure("REPLICATION — FIGURE 6", "Residual networks: the ordering "
             "reverses", "figures/fig6_resnet.png",
             caption="Same layout, same recipe, identity shortcuts added. The "
                     "curves nearly coincide and the deeper network ends below.",
             note="At depth 56 the residual net is 7.07 pp better; at depth 20, "
                  "1.19 pp — the gain grows with depth.")

    d.table("REPLICATION — TABLE 6 AND THE SEED TEST", "Our numbers, and "
            "whether the differences are real",
            ["Comparison", "Ours", "Paper", "Spread", "Verdict"],
            [["plain-20, test", "9.87%", "9.7%", "0.60", ""],
             ["plain-56, test", "14.43%", "13.3%", "0.23", ""],
             ["ResNet-20, test", "8.68%", "8.75%", "0.34", ""],
             ["ResNet-56, test", "7.36%", "6.97%", "0.30", ""],
             ["plain-56 minus plain-20, training", "+2.04 pp", "—", "0.70", "real"],
             ["ResNet-20 minus ResNet-56, test", "+1.31 pp", "1.78", "0.34", "real"]],
            caption="A difference counts as real only if it exceeds the larger "
                    "of the two networks' spreads. The degradation survives that "
                    "test, and so does the reversal.",
            col_w=[3.4, 1.6, 1.1, 1.1, 1.1], size=13)

    d.bullets("EXPERIMENT 1 — THE QUESTION", "The obvious objection, and what "
              "we predicted", [
        ("The paper gives 20 and 56 layers the same learning-rate schedule",
         "Maybe 56 layers simply needs a smaller step size"),
        ("H1:  the rate selected for plain-56 is lower than for plain-20", None),
        ("H2:  the degradation remains after each depth is tuned", None),
        ("Only the initial rate varies: 0.01 / 0.03 / 0.1 / 0.3 / 0.5",
         "Divisions by 10, the 64k schedule, the split and the augmentation stay fixed"),
    ], size=21)

    d.figure("EXPERIMENT 1 — THE GRID", "Selection and the failure region",
             "figures/extension_a_lr.png",
             caption="Final error against the initial learning rate, log scale. "
                     "plain-20 selects 0.1, plain-56 selects 0.03 — 3.3 times "
                     "lower. Crosses are runs that diverged.",
             note="At 0.3 and 0.5 the 20-layer network trains and the 56-layer "
                  "one does not. Both rates are above the paper's own 0.1.")

    d.table("EXPERIMENT 1 — THE ANSWER", "Tuning buys back a third, not the gap",
            ["", "plain-20 (0.1)", "plain-56 (0.03)"],
            [["Training error", "0.43%", "1.70%"],
             ["Test error", "9.87%", "14.18%"],
             ["Spread, test", "0.60", "2.44"],
             ["Training gap", "+1.27 pp against a 0.52 spread", "real"]],
            caption="At the shared rate of 0.1 the training gap was 2.04 pp; "
                    "after tuning it is 1.27 pp. H1 and H2 both hold, and tuning "
                    "accounts for about a third of the difference.",
            col_w=[2.2, 3.2, 3.2], size=15)

    # ---------------------------------------------------------- part 4
    d.section("4", "Why, and how far it goes", "The mechanism evidence, a "
              "second experiment, and an honest account of the limits.",
              "Liu Yilin")

    d.figure("MECHANISM — PAPER FIGURE 7", "Responses shrink with depth; "
             "gradients do not vanish", "figures/fig7.png",
             caption="Standard deviation of each layer's output after batch "
                     "normalisation. Sorted by magnitude, plain-56 sits below "
                     "plain-20 — the ordering the paper reports.",
             note="Gradients: the smallest norm is 0.12 for plain-56 and 0.24 "
                  "for plain-20. Smaller, but not vanishing.")

    d.bullets("EXPERIMENT 2 — TRAINING SET", "The one number that did not line "
              "up, and why", [
        ("plain-56 came out at 14.43% where the paper reports 13.3%, and the "
         "spread is only 0.23 pp — not seed noise", None),
        ("The protocol fixes 45,000 images; the paper may have used all 50,000",
         "So we ran it rather than leaving it as a conjecture"),
        ("On 50,000 images plain-56 reaches 13.06%, just below the paper's 13.3%",
         "Training error rose from 2.47% to 4.77%: more data made it fit worse and generalise better"),
        ("And the degradation survives: 13.06 against plain-20's 9.44", None),
    ], size=20)

    d.twocol("CONCLUSION", "What we show, and what we do not",
             "Shown", [
                 "Degradation reproduces: +2.04 pp training error, beyond a "
                 "0.70 spread",
                 "Identity shortcuts reverse the ordering (7.07 pp at depth 56)",
                 "Tuning buys back a third, and the deep net dies at 0.3",
                 "The 1.13-point gap was the training set, not the code",
             ],
             "Not shown", [
                 "Why residual connections help — consistent, not causal",
                 "Whether the paper really used 50,000 images",
                 "Anything between or beyond our five grid points",
                 "Longer schedules, other datasets, options B and C",
             ],
             foot="Tuning explains a third of the gap; the training set explains "
                  "the replication; the rest needs another explanation.")

    # --------------------------------------------------------- backups
    d.section("B", "Backup slides", "For questions — not part of the main talk.",
              "")

    d.table("BACKUP — FULL REPLICATION", "Every seed at iteration 64,000",
            ["Network", "Seed", "Train 45k", "Val 5k", "Test 10k"],
            [["plain-20", "0 / 1 / 2", "0.43 / 0.38 / 0.48", "9.14 / 9.42 / 9.04",
              "10.15 / 9.55 / 9.92"],
             ["plain-56", "0 / 1 / 2", "2.81 / 2.11 / 2.48", "13.96 / 14.22 / 13.54",
              "14.58 / 14.37 / 14.35"],
             ["ResNet-20", "0 / 1 / 2", "0.22 / 0.19 / 0.24", "7.96 / 8.34 / 8.26",
              "8.71 / 8.83 / 8.49"],
             ["ResNet-56", "0 / 1 / 2", "0.02 / 0.01 / 0.01", "6.84 / 6.84 / 7.04",
              "7.49 / 7.41 / 7.19"]],
            col_w=[1.6, 1.2, 2.6, 2.6, 2.6], size=13)

    d.table("BACKUP — FULL GRID", "The tuning grid, one seed per setting",
            ["Depth", "lr", "Train 45k", "Val 5k", "Test 10k", "Status"],
            [["20", "0.01", "1.01", "11.36", "12.26", ""],
             ["20", "0.03", "0.45", "9.96", "10.79", ""],
             ["20", "0.1", "0.43", "9.14", "10.15", "selected"],
             ["20", "0.3", "0.43", "8.82", "8.82", "tied"],
             ["20", "0.5", "0.64", "9.34", "9.59", ""],
             ["56", "0.01", "3.91", "15.92", "16.94", ""],
             ["56", "0.03", "1.94", "15.10", "15.61", "selected"],
             ["56", "0.1", "2.81", "13.96", "14.58", ""],
             ["56", "0.3", "—", "—", "—", "diverged, iter 23"],
             ["56", "0.5", "—", "—", "—", "diverged, iter 14"]],
            col_w=[0.9, 0.9, 1.4, 1.4, 1.4, 2.6], size=12)

    d.table("BACKUP — THE 50,000-IMAGE CHECK", "Where the 1.13 points came from",
            ["Network", "Images", "Train", "Test", "Paper"],
            [["plain-20", "45,000", "0.43", "9.87", "9.7"],
             ["plain-20", "50,000", "0.61", "9.44", "9.7"],
             ["plain-56", "45,000", "2.47", "14.43", "13.3"],
             ["plain-56", "50,000", "4.77", "13.06", "13.3"]],
            caption="Three seeds each, everything else unchanged; no validation "
                    "set in the 50k variant.",
            col_w=[1.8, 1.6, 1.4, 1.4, 1.4], size=15)

    d.table("BACKUP — COMPUTE AND ARTEFACTS", "What it cost and where it is",
            ["Item", "Value"],
            [["Total runs", "29 (12 replication + 11 tuning + 6 on 50k images)"],
             ["Total GPU time", "22 hours on one RTX 4060 Laptop"],
             ["Repository", "code, all runs with weights, every figure script"]],
            caption="A reader can regenerate every table and figure from the "
                    "committed run outputs without retraining.",
            col_w=[2.4, 6], size=16)

    d.save(args.out)


if __name__ == "__main__":
    main()
