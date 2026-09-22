"""Tests for utility functions and memory logging."""

import os
from unittest.mock import patch, mock_open
from repomind.utils import get_current_rss_mb, log_memory


def test_get_current_rss_mb_parses_status():
    """get_current_rss_mb should parse VmRSS from /proc/self/status in MB."""
    fake_status = "Name:\tpython\nVmRSS:\t 1048576 kB\nThreads:\t1\n"
    with patch("builtins.open", mock_open(read_data=fake_status)):
        rss = get_current_rss_mb()
        assert rss == 1024.0


def test_get_current_rss_mb_fallback_on_error():
    """get_current_rss_mb should return 0.0 if /proc/self/status cannot be read."""
    with patch("builtins.open", side_effect=FileNotFoundError):
        rss = get_current_rss_mb()
        assert rss == 0.0


def test_log_memory_formats_message(caplog):
    """log_memory should format log messages with PID, stage, and RSS."""
    with caplog.at_level("INFO", logger="repomind.utils"):
        log_memory("TEST_STAGE", "chunks=5")
        assert len(caplog.records) == 1
        record = caplog.records[0]
        assert f"pid={os.getpid()}" in record.message
        assert "stage=TEST_STAGE" in record.message
        assert "rss_mb=" in record.message
        assert "chunks=5" in record.message
