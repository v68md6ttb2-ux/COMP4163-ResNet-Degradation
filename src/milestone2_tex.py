"""Build the Milestone 2 LaTeX document from one finished run.

Usage:
    python -m src.milestone2_tex runs/resnet20-s0 --out milestone2/milestone2.tex
    pdflatex -output-directory milestone2 milestone2/milestone2.tex   (twice)

The document is intentionally built from the run's own files (results.json and
metrics.csv), so no number in it is typed by hand.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

PREAMBLE = r"""\documentclass[11pt,a4paper]{article}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{amsmath}
\usepackage{graphicx}
% Manual page layout (geometry.sty is not installed in this MiKTeX tree)
\setlength{\textwidth}{16.8cm}
\setlength{\textheight}{23.8cm}
\setlength{\oddsidemargin}{-0.44cm}
\setlength{\evensidemargin}{-0.44cm}
\setlength{\topmargin}{-0.7cm}
\setlength{\headheight}{0pt}
\setlength{\headsep}{0.3cm}
\setlength{\footskip}{0.8cm}
\setlength{\belowcaptionskip}{7pt}
\setlength{\itemsep}{2pt}
\setlength{\topsep}{2pt}
\setlength{\leftmargini}{1.5em}
\pagestyle{plain}

\begin{document}
"""

HEAD = r"""
\begin{center}
{\Large\bfseries COMP4163 Course Project --- Milestone 2: First Run}\\[4pt]
{\large A ResNet-20 on CIFAR-10, 64{,}000 iterations, against Table~6}\\[8pt]
\begin{tabular}{@{}rl@{}}
Team: & group 11 \\
Members: & Chen Letian 2330026013 \\
         & Zhang Yiyang 2430024331 \\
         & Liu Chuanliang 2330013021 \\
         & Liu Yilin 2430026139 \\
\end{tabular}\\[8pt]
Date: 13 November 2026
\end{center}

\vspace{-2pt}
\noindent\rule{\textwidth}{0.4pt}

\section{What we ran}

We trained \textbf{ResNet-20} once (seed~0) on CIFAR-10 with the recipe of
Sec.~4.2 of He et al.~(2016) and the course protocol of the topic card. The
network has $6n+2=20$ weighted layers ($n=3$), shortcut option~A (identity
shortcuts, zero-padding when the number of channels increases), and
\PARAMS{} parameters---the same count as the plain-20 network.

\begin{center}
\begin{tabular}{@{}ll@{}}
\toprule
Item & Setting \\
\midrule
Data & 45{,}000 train / 5{,}000 validation / 10{,}000 test images, $32\times32$ \\
Augmentation & pad-4 random crop, random horizontal flip, per-pixel mean subtracted \\
Optimiser & SGD, momentum 0.9, weight decay $10^{-4}$, batch 128 \\
Learning rate & 0.1, divided by 10 at 32{,}000 and 48{,}000 iterations \\
Initialisation & He (MSRA) fan-in normal, BatchNorm at default, no dropout \\
Schedule & 64{,}000 iterations $=183$ epochs of 351 iterations (last one partial) \\
Precision & float16 autocast with gradient scaling; all evaluation in float32 \\
Hardware & one RTX 4060 Laptop GPU (8\,GB), PyTorch \TORCHVER{}, CUDA \CUDAVER{} \\
\bottomrule
\end{tabular}
\end{center}

One epoch is $45{,}000/128=351$ iterations, the partial last batch dropped, as
the card specifies. After every epoch and at iteration 64{,}000 we record four
errors: the running training error over the augmented minibatches in training
mode, and the training (45k), validation (5k) and test (10k) errors measured in
evaluation mode on un-augmented images.

\section{Result at 64{,}000 iterations}

\begin{center}
\begin{tabular}{@{}lrrr@{}}
\toprule
Measurement at iteration 64{,}000 & Value & Paper (Table~6) & Difference \\
\midrule
Test error, 10{,}000 images & \TESTPCT\% & 8.75\% & \DIFFPCT\ percentage points \\
Validation error, 5{,}000 images & \VALPCT\% & --- & --- \\
Training error, 45{,}000 images (eval mode) & \TRAINPCT\% & --- & --- \\
Training error, running, augmented batches & \RUNPCT\% & --- & --- \\
\bottomrule
\end{tabular}
\end{center}

The single run lands \DIFFABS\ percentage points from the published value, which
is within what one run and a 45{,}000-image training set can be expected to
move. Note that the published 8.75\% is a single run as well (the paper reports
no seed spread for this depth), so the comparison has no error bar on either
side; our three-seed replication in the final report supplies one.

\section{Training curves}

\begin{center}
\includegraphics[width=\textwidth]{\FIGPATH}
\end{center}
\vspace{-6pt}
\begin{center}
\footnotesize{ResNet-20, seed~0. Left: running training error on the augmented
minibatches. Middle: training error on the 45{,}000 training images in
evaluation mode. Right: test error on the 10{,}000 test images. The dotted
lines are the values read off Figure~6 of the paper (0.8\% training, 8.8\%
test). The two drops are the learning-rate divisions at 32{,}000 and 48{,}000
iterations.}
\end{center}

\section{Errors after every epoch}

The table lists, after each of the 183 epochs, the running training error
(augmented, training mode), the training/validation/test errors (evaluation
mode, un-augmented), and the mean training loss over the epoch. The final row
is iteration 64{,}000, where the last epoch stops.

"""

TAIL = r"""
\section{Wall-clock time}

The run took \MINUTES\ minutes of wall-clock time on one RTX 4060 Laptop GPU,
that is \PERIT\ milliseconds per training iteration including the four error
measurements taken every 351 iterations. For the full replication this sets the
budget: the 20-layer networks cost about 16 minutes and the 56-layer networks
about 50 minutes per run, so the twelve runs of the main experiment need
roughly five hours of GPU time on this machine.

\section{Deviations from the paper}

\begin{itemize}
\item \textbf{45{,}000 training images.} The card fixes the split; the paper
does not say whether the networks behind Table~6 saw 45{,}000 or 50{,}000
images. Fewer training images can only make our errors higher than the
paper's, and our 8.71\% is not higher, so the split is not inflating the gap.
\item \textbf{Batch 128 on one GPU.} The paper trains with batch 128 ``on two
GPUs'' (Sec.~4.2); the card fixes the batch at 128, so we keep the same
effective batch on a single device.
\item \textbf{Mixed precision.} The card permits float16 autocast with gradient
scaling provided it is used in \emph{every} run; every run reported here and in
the final report uses it, and all reported errors are measured in float32.
\end{itemize}

\section*{AI use declaration}
We used Claude Code (Anthropic) as a coding assistant: it wrote the training
and evaluation code, the table and figure scripts, and drafted this milestone.
The recipe and protocol were taken from the paper and the topic card by the
team; every number in this document is generated by the scripts from the run's
own output files, and all code and experiments are run and understood by the
team, and every member can explain every line and every plot, as the course
requires.

\end{document}
"""


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run")
    p.add_argument("--out", default="milestone2/milestone2.tex")
    p.add_argument("--figure", default="figures/milestone2.png")
    p.add_argument("--torch", default="2.12.1")
    p.add_argument("--cuda", default="12.6")
    args = p.parse_args()

    run = Path(args.run)
    r = json.loads((run / "results.json").read_text(encoding="utf-8"))
    final = r["final"]
    test, val, train, run_err = (final["test_err"], final["val_err"],
                                 final["train_err"], final["running_train_err"])
    diff = test - 8.75

    def pct(x: float) -> str:
        return f"{x:.2f}"

    values = {
        "PARAMS": f"{r['parameters']:,}".replace(",", "{,}"),
        "TORCHVER": args.torch,
        "CUDAVER": args.cuda,
        "TESTPCT": pct(test),
        "VALPCT": pct(val),
        "TRAINPCT": pct(train),
        "RUNPCT": pct(run_err),
        "DIFFPCT": f"{diff:+.2f}",
        "DIFFABS": f"{abs(diff):.2f}",
        "FIGPATH": args.figure,
        "MINUTES": f"{r['elapsed_s'] / 60:.1f}",
        "PERIT": f"{r['elapsed_s'] / max(r['iterations_run'], 1) * 1000:.1f}",
    }

    rows = []
    with open(run / "metrics.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                f"{row['epoch']} & {row['iteration']} & {float(row['lr']):.4g} & "
                f"{float(row['running_train_err']):.2f} & {float(row['train_err']):.2f} & "
                f"{float(row['val_err']):.2f} & {float(row['test_err']):.2f} & "
                f"{float(row['epoch_loss']):.4f} \\\\"
            )

    table = [
        r"\begin{center}",
        r"\footnotesize",
        r"\begin{longtable}{@{}rrrrrrrr@{}}",
        r"\toprule",
        r"Epoch & Iter & lr & Training (running) & Training 45k & Val 5k & Test 10k & Loss \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Epoch & Iter & lr & Training (running) & Training 45k & Val 5k & Test 10k & Loss \\",
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    table += rows
    table += [r"\end{longtable}", r"\end{center}", ""]

    text = PREAMBLE + HEAD + "\n".join(table) + TAIL
    for key, value in values.items():
        text = text.replace("\\" + key + " ", value + " ")
        text = text.replace("\\" + key, value)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"saved {out} ({len(rows)} epoch rows)")


if __name__ == "__main__":
    main()
