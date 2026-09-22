"""Utility helpers for process monitoring and logging."""

import os
import logging
from typing import Optional

logger = logging.getLogger(__name__)


def get_current_rss_mb() -> float:
    """Get current process Resident Set Size (RSS) in MB from /proc/self/status.

    Reads /proc/self/status and parses the VmRSS field. If /proc/self/status
    is unavailable (e.g. on non-Linux systems), fails gracefully and returns 0.0.

    Returns:
        Process RSS in megabytes (MB), or 0.0 if unavailable.
    """
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        return round(float(parts[1]) / 1024.0, 2)
    except Exception:
        pass
    return 0.0


def log_memory(stage: str, extra: Optional[str] = None) -> None:
    """Log memory usage with process PID and stage.

    Emits structured log messages in the format:
    MEMORY | pid=<PID> | stage=<STAGE> | rss_mb=<MB> [| <EXTRA>]

    Args:
        stage: Current execution lifecycle stage
        extra: Optional additional key-value string (e.g., 'chunks=10')
    """
    pid = os.getpid()
    rss_mb = get_current_rss_mb()
    msg = f"MEMORY | pid={pid} | stage={stage} | rss_mb={rss_mb}"
    if extra:
        msg = f"{msg} | {extra}"
    logger.info(msg)
