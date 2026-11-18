"""
Shared helper utilities for the Smart Farm Advisory System.

Provides reusable functions for data I/O, evaluation visualisation,
logging setup, and execution timing that are consumed by every module.
"""

from __future__ import annotations

import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator, List, Optional, Sequence, Union

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server / CI environments

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.utils.config import FIGURES_DIR, LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL, LOGS_DIR

# ──────────────────────────────────────────────────────────────────────
# 1. DATA I/O HELPERS
# ──────────────────────────────────────────────────────────────────────


def load_csv(
    filepath: Union[str, Path],
    *,
    parse_dates: Optional[List[str]] = None,
    **kwargs,
) -> pd.DataFrame:
    """Load a CSV file into a :class:`~pandas.DataFrame`.

    Parameters
    ----------
    filepath:
        Absolute or relative path to the CSV file.
    parse_dates:
        Column names to parse as datetime objects.
    **kwargs:
        Extra keyword arguments forwarded to :func:`pandas.read_csv`.

    Returns
    -------
    pd.DataFrame
        The loaded data.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")

    logger = logging.getLogger(__name__)
    logger.info("Loading CSV: %s", filepath)

    df = pd.read_csv(filepath, parse_dates=parse_dates, **kwargs)
    logger.info("Loaded %d rows × %d cols from %s", df.shape[0], df.shape[1], filepath.name)
    return df


def save_csv(
    df: pd.DataFrame,
    filepath: Union[str, Path],
    *,
    index: bool = False,
    **kwargs,
) -> Path:
    """Persist a :class:`~pandas.DataFrame` to CSV.

    Parameters
    ----------
    df:
        DataFrame to save.
    filepath:
        Destination path (parent directories are created automatically).
    index:
        Whether to write the row index.  Defaults to ``False``.
    **kwargs:
        Extra keyword arguments forwarded to :meth:`DataFrame.to_csv`.

    Returns
    -------
    Path
        The resolved path of the written file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(__name__)
    df.to_csv(filepath, index=index, **kwargs)
    logger.info("Saved CSV (%d rows × %d cols) → %s", df.shape[0], df.shape[1], filepath)
    return filepath.resolve()


# ──────────────────────────────────────────────────────────────────────
# 2. EVALUATION & VISUALISATION HELPERS
# ──────────────────────────────────────────────────────────────────────


def plot_confusion_matrix(
    y_true: Sequence,
    y_pred: Sequence,
    labels: Optional[List[str]] = None,
    *,
    title: str = "Confusion Matrix",
    figsize: tuple[int, int] = (10, 8),
    cmap: str = "Blues",
    normalize: bool = False,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = 150,
) -> Path:
    """Plot and save a confusion matrix heatmap.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    labels:
        Ordered list of class label names.  If ``None`` the unique sorted
        values from *y_true* are used.
    title:
        Plot title.
    figsize:
        Figure dimensions ``(width, height)`` in inches.
    cmap:
        Matplotlib / seaborn colour map name.
    normalize:
        If ``True``, each row is normalised to sum to 1 (recall-based).
    save_path:
        Where to save the figure.  Defaults to ``FIGURES_DIR / "confusion_matrix.png"``.
    dpi:
        Dots per inch for the saved figure.

    Returns
    -------
    Path
        The resolved path of the saved figure.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred), key=str)

    cm = confusion_matrix(y_true, y_pred, labels=labels)

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        # Avoid division by zero for labels with no true samples
        row_sums[row_sums == 0] = 1
        cm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=figsize)
    fmt = ".2f" if normalize else "d"
    sns.heatmap(cm, annot=True, fmt=fmt, cmap=cmap, xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(title)
    fig.tight_layout()

    if save_path is None:
        save_path = FIGURES_DIR / "confusion_matrix.png"
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger = logging.getLogger(__name__)
    logger.info("Confusion matrix saved → %s", save_path)
    return save_path.resolve()


def print_classification_report(
    y_true: Sequence,
    y_pred: Sequence,
    *,
    target_names: Optional[List[str]] = None,
    digits: int = 4,
    output_file: Optional[Union[str, Path]] = None,
) -> str:
    """Print (and optionally save) a scikit-learn classification report.

    Parameters
    ----------
    y_true:
        Ground-truth labels.
    y_pred:
        Predicted labels.
    target_names:
        Display names for each class.
    digits:
        Number of decimal places for the metrics.
    output_file:
        If provided, the report text is also written to this file.

    Returns
    -------
    str
        The formatted classification report string.
    """
    report: str = classification_report(
        y_true, y_pred, target_names=target_names, digits=digits, zero_division=0
    )

    logger = logging.getLogger(__name__)
    logger.info("Classification Report:\n%s", report)

    if output_file is not None:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_file.write_text(report, encoding="utf-8")
        logger.info("Report saved → %s", output_file)

    return report


# ──────────────────────────────────────────────────────────────────────
# 3. LOGGING
# ──────────────────────────────────────────────────────────────────────


def setup_logging(
    name: str = "smart_farm",
    *,
    level: Optional[str] = None,
    log_to_file: bool = True,
    log_dir: Optional[Union[str, Path]] = None,
) -> logging.Logger:
    """Configure and return a named logger with console and optional file handlers.

    Parameters
    ----------
    name:
        Logger name (typically the module or package name).
    level:
        Logging level string (e.g. ``"DEBUG"``).  Defaults to the value of
        :data:`~src.utils.config.LOG_LEVEL`.
    log_to_file:
        Whether to also log to a file under *log_dir*.
    log_dir:
        Directory for log files.  Defaults to :data:`~src.utils.config.LOGS_DIR`.

    Returns
    -------
    logging.Logger
        The configured logger instance.
    """
    if level is None:
        level = LOG_LEVEL

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Prevent duplicate handlers when called multiple times
    if logger.handlers:
        return logger

    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_to_file:
        log_dir_path = Path(log_dir) if log_dir else LOGS_DIR
        log_dir_path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_dir_path / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# ──────────────────────────────────────────────────────────────────────
# 4. TIMER CONTEXT MANAGER
# ──────────────────────────────────────────────────────────────────────


@contextmanager
def Timer(description: str = "Operation") -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a code block.

    Usage::

        with Timer("Model training"):
            model.fit(X_train, y_train)
        # => INFO | Model training completed in 12.34 s

    Parameters
    ----------
    description:
        Human-readable label printed alongside the elapsed time.
    """
    logger = logging.getLogger(__name__)
    logger.info("%s — started …", description)
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if elapsed < 60:
            logger.info("%s — completed in %.2f s", description, elapsed)
        else:
            minutes, seconds = divmod(elapsed, 60)
            logger.info("%s — completed in %d min %.2f s", description, int(minutes), seconds)
