#!/usr/bin/env python
"""
Gera gráficos em estilo de publicação científica (fonte serifada, paleta
sóbria, grade discreta, alta resolução) a partir dos metrics.csv já salvos
pelo train_yolo.py e/ou train_fasterrcnn.py. Não precisa re-treinar nada.

Funciona com um único modelo ou com os dois lado a lado (sobrepostos no
mesmo gráfico, para comparação visual direta).

Uso (um modelo só):
    python plot_publication_style.py \
        --metrics-csv run1/yolo_run/metrics.csv \
        --label "YOLO11n" \
        --output-dir figuras

Uso (comparando os dois, no mesmo gráfico):
    python plot_publication_style.py \
        --metrics-csv run1/yolo_run/metrics.csv \
        --label "YOLO11n" \
        --metrics-csv2 "../Faster R-CNN/run1/fasterrcnn_run/metrics.csv" \
        --label2 "Faster R-CNN" \
        --output-dir figuras
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# --------------------------------------------------------------------------
# Estilo "publicação": fonte serifada (como LaTeX/Nature), linhas finas,
# grade discreta, fundo branco, sem moldura desnecessária nas bordas
# superior/direita, resolução alta o suficiente para impressão (300 dpi).
# --------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "lines.linewidth": 1.8,
    "lines.markersize": 3.5,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "legend.frameon": False,
    "legend.fontsize": 10,
})

COLOR_A = "#F08100"  # azul petróleo — modelo 1 (ou treino, quando 1 modelo só)
COLOR_B = "#8100B1"   # terracota — modelo 2 (ou validação, quando 1 modelo só)


def normalize_columns(df):
    """Os dois pipelines usam nomes de coluna ligeiramente diferentes
    (mAP50 vs mAP50(B), etc). Isso mapeia para um conjunto comum."""
    rename = {}
    for col in df.columns:
        c = col.strip()
        if c in ("mAP50", "metrics/mAP50(B)"):
            rename[col] = "mAP50"
        elif c in ("mAP50_95", "mAP50-95", "metrics/mAP50-95(B)"):
            rename[col] = "mAP50_95"
        elif c in ("train_loss", "train/box_loss"):
            rename[col] = "train_loss"
        elif c in ("val_loss", "val/box_loss"):
            rename[col] = "val_loss"
        elif c == "epoch":
            rename[col] = "epoch"
    return df.rename(columns=rename)


def load_metrics(path):
    df = pd.read_csv(path)
    return normalize_columns(df)


def plot_single_panel(ax, series, title, ylabel=None, ylim=None):
    """series: lista de (label, epoch_array, values_array, color) — cada
    série carrega seu PRÓPRIO eixo de épocas, já que dois modelos podem
    ter parado em números de épocas diferentes (early stopping distinto)."""
    for label, x, values, color in series:
        linestyle = "--" if "Faster R-CNN" in label else "-"
        ax.plot(x, values, color=color, label=label, linestyle=linestyle)
    ax.set_title(title)
    ax.set_xlabel("Época")
    if ylabel:
        ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    if len(series) > 1:
        ax.legend(loc="best")


def make_figures(metrics_a, label_a, metrics_b, label_b, output_dir):
    two_models = metrics_b is not None

    # --- Figura 1: curva de loss ---
    fig, ax = plt.subplots(figsize=(6.5, 4))
    if two_models:
        plot_single_panel(
            ax,
            [(f"{label_a} — treino", metrics_a["epoch"], metrics_a["train_loss"], COLOR_A),
             (f"{label_b} — treino", metrics_b["epoch"], metrics_b["train_loss"], COLOR_B)],
            "Perda de treino por época", ylabel="Loss",
        )
    else:
        plot_single_panel(
            ax,
            [("Treino", metrics_a["epoch"], metrics_a["train_loss"], COLOR_A),
             ("Validação", metrics_a["epoch"], metrics_a["val_loss"], COLOR_B)],
            f"{label_a} — curva de perda", ylabel="Loss",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "loss_curve.png")
    fig.savefig(output_dir / "loss_curve.pdf")
    plt.close(fig)

    # --- Figura 2: mAP50 e mAP50-95, lado a lado ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if two_models:
        plot_single_panel(
            axes[0],
            [(label_a, metrics_a["epoch"], metrics_a["mAP50"], COLOR_A),
             (label_b, metrics_b["epoch"], metrics_b["mAP50"], COLOR_B)],
            "mAP@0.50", ylim=(-0.02, 1.02),
        )
        plot_single_panel(
            axes[1],
            [(label_a, metrics_a["epoch"], metrics_a["mAP50_95"], COLOR_A),
             (label_b, metrics_b["epoch"], metrics_b["mAP50_95"], COLOR_B)],
            "mAP@0.50:0.95", ylim=(-0.02, None),
        )
    else:
        plot_single_panel(axes[0], [(label_a, metrics_a["epoch"], metrics_a["mAP50"], COLOR_A)], "mAP@0.50", ylim=(-0.02, 1.02))
        plot_single_panel(axes[1], [(label_a, metrics_a["epoch"], metrics_a["mAP50_95"], COLOR_A)], "mAP@0.50:0.95", ylim=(-0.02, None))
    title = f"Evolução das métricas de detecção — {label_a} vs. {label_b}" if two_models else f"Evolução das métricas de detecção — {label_a}"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(output_dir / "map_curves.png")
    fig.savefig(output_dir / "map_curves.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics-csv", type=Path, required=True)
    parser.add_argument("--label", default="Modelo")
    parser.add_argument("--metrics-csv2", type=Path, default=None, help="Opcional: segundo modelo, para sobrepor")
    parser.add_argument("--label2", default="Modelo 2")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_a = load_metrics(args.metrics_csv)
    metrics_b = load_metrics(args.metrics_csv2) if args.metrics_csv2 else None

    make_figures(metrics_a, args.label, metrics_b, args.label2, args.output_dir)

    print(f"Figuras salvas em: {args.output_dir}")
    print("Arquivos gerados: loss_curve.png/.pdf, map_curves.png/.pdf")


if __name__ == "__main__":
    main()
