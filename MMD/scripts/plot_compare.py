#!/usr/bin/env python3
"""Static SVG: per-subject dumbbell chart comparing two scored runs (results/<a>/by_subject.csv vs results/<b>/).

    python scripts/plot_compare.py raw best docs/figures/baseline_vs_best.svg --labels 베이스라인,best

Stdlib only; the SVG embeds no fonts and uses a light palette, so it renders the same in GitHub markdown,
a PDF export, or a browser. Rows are sorted by change (best − baseline).
"""
import argparse
import csv
import os
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
S1, S2, INK, INK2, MUTED, GRID, BAND, GOOD, BAD, SURF = "#2a78d6", "#eb6834", "#14181d", "#4b5563", "#8a919c", "#e3e7ed", "#f4f6f9", "#006300", "#d03b3b", "#ffffff"


def load(name):
    path = name if name.endswith(".csv") else os.path.join(ROOT, "results", name, "by_subject.csv")
    return {r["subject"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("out")
    ap.add_argument("--labels", default="baseline,best")
    ap.add_argument("--per_subject", type=int, default=30)
    args = ap.parse_args()
    la, lb = args.labels.split(",")
    A, B = load(args.a), load(args.b)
    rows = sorted(A, key=lambda s: (-(int(B[s]["pipeline_correct"]) - int(A[s]["pipeline_correct"])), -int(B[s]["pipeline_correct"])))
    ta, tb = sum(int(A[s]["pipeline_correct"]) for s in rows), sum(int(B[s]["pipeline_correct"]) for s in rows)
    n = len(rows) * args.per_subject

    W, labelW, right, rowH, top = 900, 250, 70, 24, 56
    H = top + len(rows) * rowH + 30
    plotW = W - labelW - right
    x = lambda v: labelW + v / args.per_subject * plotW
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" font-family="Noto Sans KR, Apple SD Gothic Neo, Helvetica, Arial, sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="{SURF}"/>',
         f'<text x="16" y="22" font-size="15" font-weight="700" fill="{INK}">과목별 정답 수 · {escape(la)} → {escape(lb)}</text>',
         f'<text x="16" y="40" fill="{INK2}">{escape(la)} {ta}/{n} = {100*ta/n:.2f}%   {escape(lb)} {tb}/{n} = {100*tb/n:.2f}%   변화 {tb-ta:+d}문항 ({100*(tb-ta)/n:+.2f}%p)</text>',
         f'<circle cx="{W-250}" cy="36" r="5" fill="{S1}"/><text x="{W-240}" y="40" fill="{INK2}">{escape(la)}</text>',
         f'<circle cx="{W-150}" cy="36" r="5" fill="{S2}"/><text x="{W-140}" y="40" fill="{INK2}">{escape(lb)}</text>']
    for t in range(0, args.per_subject + 1, 5):
        o.append(f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="{top-6}" y2="{H-22}" stroke="{GRID}"/>')
        o.append(f'<text x="{x(t):.1f}" y="{H-6}" text-anchor="middle" fill="{MUTED}" font-size="11">{t}</text>')
    for i, s in enumerate(rows):
        a, b = int(A[s]["pipeline_correct"]), int(B[s]["pipeline_correct"]); d = b - a
        y = top + i * rowH + rowH / 2
        if i % 2 == 0:
            o.append(f'<rect x="0" y="{y-rowH/2:.1f}" width="{W}" height="{rowH}" fill="{BAND}"/>')
        o.append(f'<text x="{labelW-12}" y="{y+4:.1f}" text-anchor="end" fill="{INK2}">{escape(s.replace("_", " "))}</text>')
        if d:
            o.append(f'<line x1="{x(a):.1f}" x2="{x(b):.1f}" y1="{y:.1f}" y2="{y:.1f}" stroke="{S2 if d>0 else S1}" stroke-opacity="0.35" stroke-width="3" stroke-linecap="round"/>')
        o.append(f'<circle cx="{x(a):.1f}" cy="{y:.1f}" r="5.5" fill="{S1}" stroke="{SURF}" stroke-width="2"/>')
        o.append(f'<circle cx="{x(b):.1f}" cy="{y:.1f}" r="5.5" fill="{S2}" stroke="{SURF}" stroke-width="2"/>')
        col = GOOD if d > 0 else BAD if d < 0 else MUTED
        o.append(f'<text x="{W-right+12}" y="{y+4:.1f}" fill="{col}" font-weight="{600 if d else 400}">{"±0" if d == 0 else f"{d:+d}"}</text>')
    o.append("</svg>")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(o) + "\n")
    print(f"wrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
