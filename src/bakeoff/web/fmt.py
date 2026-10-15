"""Display formatting. Every rounding for the site happens here; None always shows as a dash."""

DASH = "—"
MINUS = "−"


def pct(x: float | None) -> str:
    """A 0..1 fraction as a percentage with one decimal place: 0.87 -> '87.0%'."""
    return DASH if x is None else f"{x * 100:.1f}%"


def pm(ci: list[float] | tuple[float, float] | None) -> str:
    """Half-width of a 0..1 interval in percentage points: [0.8, 0.94] -> '±7.0'."""
    if ci is None:
        return DASH
    return f"±{(ci[1] - ci[0]) / 2 * 100:.1f}"


def tps(x: float | None) -> str:
    return DASH if x is None else f"{x:.1f}"


def ms(seconds: float | None) -> str:
    return DASH if seconds is None else f"{seconds * 1000:.0f} ms"


def kctx(n: int | float | None) -> str:
    """Context length in K (1024 tokens): 32768 -> '32K'."""
    return DASH if n is None else f"{n / 1024:.0f}K"


def gb(n: int | float | None) -> str:
    """Bytes in GB (10^9) with one decimal place."""
    return DASH if n is None else f"{n / 1e9:.1f} GB"


def pts(x: float | None) -> str:
    """A signed difference of 0..1 scores in percentage points: 0.28 -> '+28.0'."""
    if x is None:
        return DASH
    v = round(x * 100, 1)
    return f"+{v:.1f}" if v >= 0 else f"{MINUS}{-v:.1f}"
