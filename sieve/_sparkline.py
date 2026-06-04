"""Unicode sparkline helper.

One-line visualisation of a numeric sequence. Used to show the
baseline's context growth per turn — the curve that climbs from a few
hundred tokens to several thousand as conversation history
accumulates. Sieve's version stays flat, which is the point.
"""

from __future__ import annotations


_BARS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[int] | list[float]) -> str:
    """Render a numeric sequence as a compact Unicode bar sparkline.

    Empty or all-zero input returns an empty string. Values are
    normalised to the 0..7 bar range against the max; the minimum is
    anchored at zero so the curve shows absolute growth, not deviation
    from the minimum. Negative values are clipped to zero.
    """
    nums = [max(0.0, float(v)) for v in (values or [])]
    if not nums:
        return ""
    hi = max(nums)
    if hi <= 0:
        return _BARS[0] * len(nums)
    out: list[str] = []
    scale = len(_BARS) - 1
    for v in nums:
        idx = int(round(v / hi * scale))
        idx = max(0, min(scale, idx))
        out.append(_BARS[idx])
    return "".join(out)
