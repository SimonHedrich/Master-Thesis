"""Root logger configuration for the YOLOv5s training pipeline.

Console output is routed through `tqdm.write()` so log lines don't scribble
over active progress bars. A second handler mirrors everything to a file
which the entry point uploads to MLflow at run end.
"""
from __future__ import annotations

import logging
from pathlib import Path

from tqdm import tqdm

_FORMAT = "%(asctime)s %(levelname)-5s %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_NOISY_LIBS = ("PIL", "matplotlib", "urllib3", "mlflow.utils", "mlflow.tracking")


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record))
        except Exception:
            self.handleError(record)


def setup_logging(log_file: Path | None = None, level: int = logging.INFO) -> Path | None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = TqdmLoggingHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    resolved: Path | None = None
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        resolved = log_file

    root.setLevel(level)

    for name in _NOISY_LIBS:
        logging.getLogger(name).setLevel(logging.WARNING)

    return resolved
