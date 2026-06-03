import csv
import sys
from pathlib import Path


def cell_color(value, lo, hi):
    if value is None:
        return "#d9d9d9"
    t = 0 if hi == lo else (value - lo) / (hi - lo)
    stops = [(49, 54, 149), (69, 173, 168), (255, 236, 120)]
    if t < 0.5:
        a, b, u = stops[0], stops[1], t / 0.5
    else:
        a, b, u = stops[1], stops[2], (t - 0.5) / 0.5
    return "#%02x%02x%02x" % tuple(
        round(a[i] + (b[i] - a[i]) * u) for i in range(3)
    )


def main():
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python plot_q3_heatmap.py q3_optimized_sweep.csv out.svg")

    csv_path = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    rows = list(csv.DictReader(csv_path.open()))

    bms = sorted({int(r["bm"]) for r in rows})
    bhs = sorted({int(r["bh"]) for r in rows})
    values = {}
    for row in rows:
        key = (int(row["bm"]), int(row["bh"]))
        values[key] = None if row["status"] != "ok" else float(row["tflops"])

    valid_values = [v for v in values.values() if v is not None]
    lo, hi = min(valid_values), max(valid_values)
    best_key = max((k for k, v in values.items() if v is not None), key=lambda k: values[k])

    cell_w, cell_h = 104, 44
    left, top = 88, 96
    right, bottom = 160, 70
    width = left + len(bhs) * cell_w + right
    height = top + len(bms) * cell_h + bottom

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,Helvetica,sans-serif;fill:#202124}.title{font-size:18px;font-weight:700}.axis{font-size:13px}.tick{font-size:12px}.cell{font-size:12px;font-weight:700}.small{font-size:11px}</style>',
        f'<text class="title" x="{width / 2}" y="28" text-anchor="middle">Q3 Optimized Fused MLP TFLOPs Heatmap</text>',
        f'<text class="axis" x="{left + len(bhs) * cell_w / 2}" y="58" text-anchor="middle">BH</text>',
        f'<text class="axis" x="22" y="{top + len(bms) * cell_h / 2}" text-anchor="middle" transform="rotate(-90 22 {top + len(bms) * cell_h / 2})">BM</text>',
    ]

    for j, bh in enumerate(bhs):
        x = left + j * cell_w + cell_w / 2
        parts.append(f'<text class="tick" x="{x}" y="{top - 16}" text-anchor="middle">{bh}</text>')
    for i, bm in enumerate(bms):
        y = top + i * cell_h + cell_h / 2 + 4
        parts.append(f'<text class="tick" x="{left - 14}" y="{y}" text-anchor="end">{bm}</text>')

    for i, bm in enumerate(bms):
        for j, bh in enumerate(bhs):
            x = left + j * cell_w
            y = top + i * cell_h
            value = values[(bm, bh)]
            stroke = "#111111" if (bm, bh) == best_key else "#ffffff"
            stroke_width = 2.5 if (bm, bh) == best_key else 1
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w}" height="{cell_h}" fill="{cell_color(value, lo, hi)}" stroke="{stroke}" stroke-width="{stroke_width}"/>'
            )
            label = "OOM" if value is None else f"{value:.1f}"
            fill = (
                "#202124"
                if value is None or (value is not None and (value - lo) / (hi - lo) > 0.62)
                else "#ffffff"
            )
            parts.append(
                f'<text class="cell" x="{x + cell_w / 2}" y="{y + cell_h / 2 + 4}" text-anchor="middle" fill="{fill}">{label}</text>'
            )

    legend_x = left + len(bhs) * cell_w + 34
    legend_y = top
    legend_h = len(bms) * cell_h
    for step in range(80):
        value = hi - (hi - lo) * step / 79
        y = legend_y + step * legend_h / 80
        parts.append(
            f'<rect x="{legend_x}" y="{y:.2f}" width="18" height="{legend_h / 80 + 0.8:.2f}" fill="{cell_color(value, lo, hi)}"/>'
        )
    parts.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="18" height="{legend_h}" fill="none" stroke="#555" stroke-width="1"/>',
            f'<text class="small" x="{legend_x + 28}" y="{legend_y + 4}" dominant-baseline="hanging">{hi:.1f}</text>',
            f'<text class="small" x="{legend_x + 28}" y="{legend_y + legend_h}" dominant-baseline="ideographic">{lo:.1f}</text>',
            f'<text class="small" x="{legend_x}" y="{legend_y + legend_h + 20}">TFLOPs</text>',
            f'<text class="small" x="{left}" y="{height - 22}">Best: BM={best_key[0]}, BH={best_key[1]}, {values[best_key]:.2f} TFLOPs</text>',
            "</svg>",
        ]
    )

    out_path.write_text("\n".join(parts) + "\n")


if __name__ == "__main__":
    main()
