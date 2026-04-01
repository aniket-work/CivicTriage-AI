"""ASCII summaries for terminal output."""

from __future__ import annotations

def ascii_table(
    headers: list[str],
    rows: list[list[str | float]],
) -> str:
    str_rows: list[list[str]] = [
        [str(h) for h in headers],
    ]
    for r in rows:
        str_rows.append([f"{c:.4f}" if isinstance(c, float) else str(c) for c in r])
    widths = [max(len(row[i]) for row in str_rows) for i in range(len(headers))]
    lines: list[str] = []
    sep = "+-" + "-+-".join("-" * w for w in widths) + "-+"
    lines.append(sep)

    def fmt_row(cells: list[str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(cells, widths)) + " |"

    lines.append(fmt_row(str_rows[0]))
    lines.append(sep)
    for row in str_rows[1:]:
        lines.append(fmt_row(row))
    lines.append(sep)
    return "\n".join(lines)


def summarize_run(
    phase: str,
    metrics: dict[str, float],
    n_train: int,
    n_prefs: int,
) -> str:
    rows = [
        [phase, metrics["accuracy"], metrics["macro_f1"], n_train, n_prefs],
    ]
    return ascii_table(
        ["phase", "accuracy", "macro_f1", "train_rows", "preference_rows"],
        rows,
    )
