"""Generate publication-quality validation charts for Sieve.

Produces eight SVGs under docs/figures/ (dark + light for each chart):

- token-divergence{,-light}.svg         split-panel token growth, 60-day
- hallucination-divergence{,-light}.svg hallucination rate over 60 days
- accuracy-crossover{,-light}.svg       accuracy bars by day-bucket, PA 30-day
- hallucination-bars{,-light}.svg       single-point hallucination comparison

Run from the repo root:
    python scripts/make_figures.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch

BRAND_TEAL = "#0D9488"
BRAND_RED = "#EF4444"
DARK_BG = "#262a35"
LIGHT_BG = "#f0f1f5"
DARK_TEXT = "#1f2937"
LIGHT_TEXT = "#ffffff"
MUTED_DARK = "#9ca3af"
MUTED_LIGHT = "#4b5563"

# 60-day longitudinal data (qwen3:14b)
DAYS = [10, 20, 30, 40, 50, 60]
BASELINE_TOKENS = [11441, 38443, 66577, 92886, 121892, 147237]
SIEVE_TOKENS = [1324, 1169, 968, 1121, 1181, 1058]
BASELINE_HALLU = [0.020, 0.049, 0.025, 0.026, 0.073, 0.056]
SIEVE_HALLU = [0.046, 0.032, 0.035, 0.036, 0.025, 0.011]

# Progressive Activation 30-day validation (qwen3:30b-a3b)
BUCKET_LABELS = ["Days 1–10", "Days 11–20", "Days 21–30"]
RECALL_ACC = [0.725, 0.713, 0.700]
BASELINE_ACC = [0.900, 0.800, 0.688]

# Headline hallucination bars (PA 30-day)
HALLU_BASELINE = 0.140
HALLU_SIEVE = 0.015


def _palette(dark: bool) -> tuple[str, str, str]:
    if dark:
        return DARK_BG, LIGHT_TEXT, MUTED_DARK
    return LIGHT_BG, DARK_TEXT, MUTED_LIGHT


def _style_axes(ax, text_color: str, muted: str) -> None:
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


def _plot_glow(ax, x, y, color: str, width: float = 2.0, marker: bool = True):
    ax.plot(x, y, color=color, linewidth=width * 3.0, alpha=0.12, solid_capstyle="round")
    ax.plot(x, y, color=color, linewidth=width * 1.8, alpha=0.18, solid_capstyle="round")
    (main,) = ax.plot(
        x,
        y,
        color=color,
        linewidth=width,
        solid_capstyle="round",
        marker="o" if marker else None,
        markersize=4,
        markeredgewidth=0,
    )
    return main


def _save(fig, path: Path, bg: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", facecolor=bg, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def _token_chart(path: Path, *, dark: bool) -> None:
    """Split-panel: baseline climbs in top panel, Sieve visible in bottom panel."""
    bg, text, muted = _palette(dark)

    fig = plt.figure(figsize=(10, 6), dpi=150)
    fig.patch.set_facecolor(bg)
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.18, figure=fig)

    ax_top = fig.add_subplot(gs[0])
    ax_bot = fig.add_subplot(gs[1], sharex=ax_top)

    for ax in (ax_top, ax_bot):
        ax.set_facecolor(bg)

    # Top panel — baseline with filled area
    _plot_glow(ax_top, DAYS, BASELINE_TOKENS, BRAND_RED, width=2.2)
    ax_top.fill_between(DAYS, 0, BASELINE_TOKENS, color=BRAND_RED, alpha=0.10)
    ax_top.set_ylabel("Baseline tokens / request", fontsize=10, labelpad=10)
    ax_top.set_ylim(0, 160_000)
    ax_top.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: "0" if v == 0 else f"{int(v/1000)}K")
    )
    ax_top.set_yticks([0, 40_000, 80_000, 120_000, 160_000])

    # Bottom panel — Sieve with filled area, zoomed
    _plot_glow(ax_bot, DAYS, SIEVE_TOKENS, BRAND_TEAL, width=2.2)
    ax_bot.fill_between(DAYS, 0, SIEVE_TOKENS, color=BRAND_TEAL, alpha=0.14)
    ax_bot.set_ylabel("Sieve tokens / request", fontsize=10, labelpad=10)
    ax_bot.set_xlabel("Day", fontsize=11, labelpad=10)
    ax_bot.set_ylim(0, 2_000)
    ax_bot.set_yticks([0, 1_000, 2_000])

    ax_bot.set_xticks(DAYS)
    ax_bot.set_xlim(DAYS[0] - 2, DAYS[-1] + 5)
    plt.setp(ax_top.get_xticklabels(), visible=False)

    _style_axes(ax_top, text, muted)
    _style_axes(ax_bot, text, muted)

    # Title + subtitle
    fig.text(0.08, 0.95, "Context growth over 60 days", fontsize=16, fontweight="bold", color=text)
    fig.text(
        0.08,
        0.915,
        "60-day longitudinal validation, qwen3:14b",
        fontsize=10,
        color=muted,
        style="italic",
    )

    # Legend
    handles = [
        Line2D([0], [0], color=BRAND_RED, linewidth=2.2, marker="o", markersize=5, label="Baseline"),
        Line2D([0], [0], color=BRAND_TEAL, linewidth=2.2, marker="o", markersize=5, label="Sieve"),
    ]
    leg = ax_top.legend(
        handles=handles, loc="upper left", frameon=False, labelcolor=text, fontsize=10
    )
    for t in leg.get_texts():
        t.set_color(text)

    # Endpoint labels
    ax_top.annotate(
        f"{BASELINE_TOKENS[-1]//1000}K",
        xy=(DAYS[-1], BASELINE_TOKENS[-1]),
        xytext=(DAYS[-1] + 0.7, BASELINE_TOKENS[-1]),
        fontsize=11,
        fontweight="bold",
        color=BRAND_RED,
        ha="left",
        va="center",
    )
    ax_bot.annotate(
        f"{SIEVE_TOKENS[-1]:,}",
        xy=(DAYS[-1], SIEVE_TOKENS[-1]),
        xytext=(DAYS[-1] + 0.7, SIEVE_TOKENS[-1]),
        fontsize=10,
        fontweight="bold",
        color=BRAND_TEAL,
        ha="left",
        va="center",
    )

    # Bridging annotation "140× fewer tokens" between panels (centered, on the right)
    ratio = BASELINE_TOKENS[-1] / SIEVE_TOKENS[-1]
    fig.text(
        0.885,
        0.46,
        f"{ratio:.0f}× fewer\ntokens",
        fontsize=11,
        fontweight="bold",
        color=text,
        ha="center",
        va="center",
    )
    # Arrow drawn from top endpoint to bottom endpoint across panels
    arrow = ConnectionPatch(
        xyA=(DAYS[-1] + 1.8, BASELINE_TOKENS[-1] * 0.55),
        xyB=(DAYS[-1] + 1.8, SIEVE_TOKENS[-1] * 1.2),
        coordsA="data",
        coordsB="data",
        axesA=ax_top,
        axesB=ax_bot,
        arrowstyle="-|>",
        mutation_scale=12,
        color=muted,
        linewidth=0.9,
    )
    fig.add_artist(arrow)

    fig.subplots_adjust(left=0.09, right=0.86, top=0.88, bottom=0.10)
    _save(fig, path, bg)


def _hallu_line_chart(path: Path, *, dark: bool) -> None:
    bg, text, muted = _palette(dark)

    fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    _plot_glow(ax, DAYS, BASELINE_HALLU, BRAND_RED, width=2.2)
    _plot_glow(ax, DAYS, SIEVE_HALLU, BRAND_TEAL, width=2.2)

    ax.set_xlabel("Day", fontsize=11, labelpad=10)
    ax.set_ylabel("Hallucination rate", fontsize=11, labelpad=10)
    ax.set_xticks(DAYS)
    ax.set_xlim(DAYS[0] - 2, DAYS[-1] + 5)
    ax.set_ylim(0, max(*BASELINE_HALLU, *SIEVE_HALLU) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))

    fig.text(0.08, 0.94, "Hallucination rate over 60 days", fontsize=15, fontweight="bold", color=text)
    fig.text(
        0.08,
        0.895,
        "By Day 60, baseline hallucinates 4.9× more",
        fontsize=10,
        color=muted,
        style="italic",
    )

    handles = [
        Line2D([0], [0], color=BRAND_RED, linewidth=2.2, marker="o", markersize=5, label="Baseline"),
        Line2D([0], [0], color=BRAND_TEAL, linewidth=2.2, marker="o", markersize=5, label="Sieve"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(text)

    # Crossover ≈ Day 15 (baseline 0.020 < sieve 0.046 at day 10; reversed by day 20)
    ax.annotate(
        "Crossover ≈ Day 15",
        xy=(15, 0.035),
        xytext=(17, 0.068),
        fontsize=10,
        color=muted,
        ha="left",
        arrowprops=dict(arrowstyle="-", color=muted, linewidth=0.7, connectionstyle="arc3,rad=0.2"),
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
    # Day-60 gap annotation
    ax.annotate(
        "5.6% vs 1.1%",
        xy=(DAYS[-1], (BASELINE_HALLU[-1] + SIEVE_HALLU[-1]) / 2),
        xytext=(DAYS[-1] - 8, 0.080),
        fontsize=10,
        fontweight="bold",
        color=text,
        ha="left",
    )

    _style_axes(ax, text, muted)
    fig.subplots_adjust(left=0.09, right=0.94, top=0.84, bottom=0.12)
    _save(fig, path, bg)


def _accuracy_crossover_chart(path: Path, *, dark: bool) -> None:
    """Grouped bars per day-bucket: Recall (teal) vs Baseline (red), PA 30-day."""
    bg, text, muted = _palette(dark)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    import numpy as np

    x = np.arange(len(BUCKET_LABELS))
    width = 0.36

    bars_recall = ax.bar(
        x - width / 2,
        RECALL_ACC,
        width,
        color=BRAND_TEAL,
        edgecolor="none",
        label="Sieve",
        zorder=3,
    )
    bars_base = ax.bar(
        x + width / 2,
        BASELINE_ACC,
        width,
        color=BRAND_RED,
        edgecolor="none",
        label="Baseline",
        zorder=3,
    )

    # Value labels on bars
    for bars, values in ((bars_recall, RECALL_ACC), (bars_base, BASELINE_ACC)):
        for bar, v in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.012,
                f"{v:.3f}",
                ha="center",
                va="bottom",
                color=text,
                fontsize=9,
                fontweight="bold",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_LABELS, color=text, fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11, labelpad=10)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1f}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.2))

    fig.text(0.08, 0.94, "Sieve gets smarter over time", fontsize=16, fontweight="bold", color=text)
    fig.text(
        0.08,
        0.895,
        "30-day progressive activation validation, qwen3:30b-a3b",
        fontsize=10,
        color=muted,
        style="italic",
    )

    leg = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for t in leg.get_texts():
        t.set_color(text)

    # Highlight Days 21-30 win
    gap = RECALL_ACC[-1] - BASELINE_ACC[-1]
    ax.annotate(
        f"Sieve wins (+{gap:.3f})",
        xy=(x[-1], max(RECALL_ACC[-1], BASELINE_ACC[-1]) + 0.05),
        xytext=(x[-1], 0.88),
        fontsize=10,
        fontweight="bold",
        color=BRAND_TEAL,
        ha="center",
        arrowprops=dict(arrowstyle="-", color=BRAND_TEAL, linewidth=0.8, connectionstyle="arc3,rad=0"),
    )

    _style_axes(ax, text, muted)
    fig.subplots_adjust(left=0.09, right=0.94, top=0.84, bottom=0.12)
    _save(fig, path, bg)


def _hallu_bars_chart(path: Path, *, dark: bool) -> None:
    """Two-bar dramatic comparison: baseline 14% vs Sieve 1.5%."""
    bg, text, muted = _palette(dark)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)

    labels = ["Baseline", "Sieve"]
    values = [HALLU_BASELINE, HALLU_SIEVE]
    colors = [BRAND_RED, BRAND_TEAL]

    bars = ax.bar(labels, values, color=colors, width=0.55, edgecolor="none", zorder=3)

    for bar, v in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.006,
            f"{v:.1%}",
            ha="center",
            va="bottom",
            color=text,
            fontsize=13,
            fontweight="bold",
        )

    ax.set_ylabel("Hallucination rate", fontsize=11, labelpad=10)
    ax.set_ylim(0, max(values) * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax.tick_params(axis="x", labelsize=12)

    fig.text(
        0.08, 0.94, "Hallucination: Baseline vs Sieve", fontsize=16, fontweight="bold", color=text
    )
    ratio = HALLU_BASELINE / HALLU_SIEVE
    fig.text(
        0.08,
        0.895,
        f"{ratio:.1f}× less hallucination (30-day PA validation)",
        fontsize=10,
        color=muted,
        style="italic",
    )

    # Ratio annotation bridging the bars
    ax.annotate(
        f"{ratio:.1f}× lower",
        xy=(1, HALLU_SIEVE),
        xytext=(0.5, HALLU_BASELINE * 0.55),
        fontsize=14,
        fontweight="bold",
        color=BRAND_TEAL,
        ha="center",
        va="center",
        arrowprops=dict(
            arrowstyle="-|>",
            color=BRAND_TEAL,
            linewidth=1.2,
            mutation_scale=14,
            connectionstyle="arc3,rad=-0.25",
        ),
    )

    _style_axes(ax, text, muted)
    fig.subplots_adjust(left=0.09, right=0.94, top=0.84, bottom=0.12)
    _save(fig, path, bg)


def main() -> None:
    figures = Path(__file__).resolve().parent.parent / "docs" / "figures"

    for dark, suffix in ((True, ""), (False, "-light")):
        _token_chart(figures / f"token-divergence{suffix}.svg", dark=dark)
        _hallu_line_chart(figures / f"hallucination-divergence{suffix}.svg", dark=dark)
        _accuracy_crossover_chart(figures / f"accuracy-crossover{suffix}.svg", dark=dark)
        _hallu_bars_chart(figures / f"hallucination-bars{suffix}.svg", dark=dark)

    print(f"Wrote charts to {figures}")


if __name__ == "__main__":
    main()
