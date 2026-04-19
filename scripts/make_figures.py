"""Generate publication-quality validation charts for Sieve.

Produces three SVGs under docs/figures/:
- token-divergence.svg         (dark background, for dark README mode)
- token-divergence-light.svg   (light background, for light README mode)
- hallucination-divergence.svg (dark background)

Run from the repo root:
    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

BRAND_TEAL = "#0D9488"
BRAND_RED = "#EF4444"
DARK_BG = "#262a35"
LIGHT_BG = "#f0f1f5"
DARK_TEXT = "#1f2937"
LIGHT_TEXT = "#ffffff"
MUTED_DARK = "#9ca3af"
MUTED_LIGHT = "#4b5563"

DAYS = [10, 20, 30, 40, 50, 60]
BASELINE_TOKENS = [11441, 38443, 66577, 92886, 121892, 147237]
SIEVE_TOKENS = [1324, 1169, 968, 1121, 1181, 1058]
BASELINE_HALLU = [0.020, 0.049, 0.025, 0.026, 0.073, 0.056]
SIEVE_HALLU = [0.046, 0.032, 0.035, 0.036, 0.025, 0.011]


def _style_axes(ax, text_color: str, muted: str) -> None:
    """Minimal, clean, scientific styling."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(muted)
        ax.spines[spine].set_linewidth(0.8)
    ax.tick_params(colors=text_color, labelsize=10, length=4, width=0.8)
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    ax.grid(False)


def _plot_glow(ax, x, y, color: str, width: float = 2.0) -> Line2D:
    """Main line with a subtle glow (wide low-alpha line behind)."""
    ax.plot(x, y, color=color, linewidth=width * 3.0, alpha=0.12, solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=width * 1.8, alpha=0.18, solid_capstyle="round")
    (main,) = ax.plot(
        x,
        y,
        color=color,
        linewidth=width,
        solid_capstyle="round",
        marker="o",
        markersize=4,
        markeredgewidth=0,
    )
    return main


def _token_chart(path: Path, *, dark: bool) -> None:
    bg = DARK_BG if dark else LIGHT_BG
    text = LIGHT_TEXT if dark else DARK_TEXT
    muted = MUTED_DARK if dark else MUTED_LIGHT

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    _plot_glow(ax, DAYS, BASELINE_TOKENS, BRAND_RED, width=2.2)
    _plot_glow(ax, DAYS, SIEVE_TOKENS, BRAND_TEAL, width=2.2)

    ax.set_xlabel("Day", fontsize=11, labelpad=10)
    ax.set_ylabel("Tokens per request", fontsize=11, labelpad=10)
    ax.set_xticks(DAYS)
    ax.set_xlim(DAYS[0] - 2, DAYS[-1] + 4)
    ax.set_ylim(0, max(BASELINE_TOKENS) * 1.10)

    # Y-axis formatting: thousands with K suffix
    ax.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{int(v/1000)}K")
    )

    # Title + subtitle (manual, matplotlib suptitle crowds the axes)
    fig.text(
        0.08,
        0.94,
        "Context growth over 60 days — Baseline vs Sieve",
        fontsize=15,
        fontweight="bold",
        color=text,
    )
    fig.text(
        0.08,
        0.895,
        "60-day longitudinal validation on qwen3:14b with cross-family grading",
        fontsize=10,
        color=muted,
        style="italic",
    )

    # Legend (manual, top-right)
    legend_handles = [
        Line2D([0], [0], color=BRAND_RED, linewidth=2.2, marker="o", markersize=5, label="Baseline"),
        Line2D([0], [0], color=BRAND_TEAL, linewidth=2.2, marker="o", markersize=5, label="Sieve"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        labelcolor=text,
        fontsize=10,
    )
    for t in leg.get_texts():
        t.set_color(text)

    # Day-60 gap annotation
    ax.annotate(
        "147K vs 1K",
        xy=(DAYS[-1], BASELINE_TOKENS[-1]),
        xytext=(DAYS[-1] + 0.5, BASELINE_TOKENS[-1] * 0.72),
        fontsize=11,
        fontweight="bold",
        color=BRAND_RED,
        ha="left",
        va="center",
        arrowprops=dict(
            arrowstyle="-",
            color=muted,
            linewidth=0.7,
            connectionstyle="arc3,rad=-0.15",
        ),
    )
    # Sieve endpoint label
    ax.annotate(
        f"{SIEVE_TOKENS[-1]}",
        xy=(DAYS[-1], SIEVE_TOKENS[-1]),
        xytext=(DAYS[-1] + 0.7, SIEVE_TOKENS[-1] + max(BASELINE_TOKENS) * 0.035),
        fontsize=10,
        fontweight="bold",
        color=BRAND_TEAL,
        ha="left",
    )

    _style_axes(ax, text, muted)
    fig.subplots_adjust(left=0.09, right=0.94, top=0.84, bottom=0.12)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor=bg, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _hallu_chart(path: Path) -> None:
    bg = DARK_BG
    text = LIGHT_TEXT
    muted = MUTED_DARK

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    _plot_glow(ax, DAYS, BASELINE_HALLU, BRAND_RED, width=2.2)
    _plot_glow(ax, DAYS, SIEVE_HALLU, BRAND_TEAL, width=2.2)

    ax.set_xlabel("Day", fontsize=11, labelpad=10)
    ax.set_ylabel("Hallucination rate", fontsize=11, labelpad=10)
    ax.set_xticks(DAYS)
    ax.set_xlim(DAYS[0] - 2, DAYS[-1] + 4)
    ax.set_ylim(0, max(*BASELINE_HALLU, *SIEVE_HALLU) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.2%}"))

    fig.text(
        0.08,
        0.94,
        "Hallucination rate over 60 days",
        fontsize=15,
        fontweight="bold",
        color=text,
    )
    fig.text(
        0.08,
        0.895,
        "By Day 60, baseline hallucinates 4.9× more",
        fontsize=10,
        color=muted,
        style="italic",
    )

    legend_handles = [
        Line2D([0], [0], color=BRAND_RED, linewidth=2.2, marker="o", markersize=5, label="Baseline"),
        Line2D([0], [0], color=BRAND_TEAL, linewidth=2.2, marker="o", markersize=5, label="Sieve"),
    ]
    leg = ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=10,
    )
    for t in leg.get_texts():
        t.set_color(text)

    # Crossover annotation around day 15 — where the lines cross
    # (Day 10: baseline 0.020 < sieve 0.046. Day 20: baseline 0.049 > sieve 0.032. Cross ~15.)
    ax.annotate(
        "Crossover ≈ Day 15",
        xy=(15, 0.035),
        xytext=(17, 0.068),
        fontsize=10,
        color=muted,
        ha="left",
        arrowprops=dict(
            arrowstyle="-",
            color=muted,
            linewidth=0.7,
            connectionstyle="arc3,rad=0.2",
        ),
    )
    # Day-60 endpoint labels
    ax.annotate(
        f"{BASELINE_HALLU[-1]:.1%}",
        xy=(DAYS[-1], BASELINE_HALLU[-1]),
        xytext=(DAYS[-1] + 0.7, BASELINE_HALLU[-1]),
        fontsize=10,
        fontweight="bold",
        color=BRAND_RED,
        ha="left",
        va="center",
    )
    ax.annotate(
        f"{SIEVE_HALLU[-1]:.1%}",
        xy=(DAYS[-1], SIEVE_HALLU[-1]),
        xytext=(DAYS[-1] + 0.7, SIEVE_HALLU[-1]),
        fontsize=10,
        fontweight="bold",
        color=BRAND_TEAL,
        ha="left",
        va="center",
    )

    _style_axes(ax, text, muted)
    fig.subplots_adjust(left=0.09, right=0.94, top=0.84, bottom=0.12)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor=bg, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    figures = Path(__file__).resolve().parent.parent / "docs" / "figures"
    _token_chart(figures / "token-divergence.svg", dark=True)
    _token_chart(figures / "token-divergence-light.svg", dark=False)
    _hallu_chart(figures / "hallucination-divergence.svg")
    print(f"Wrote charts to {figures}")


if __name__ == "__main__":
    main()
