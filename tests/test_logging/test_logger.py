"""
Tests for logging/utils.py
"""
import pytest
import logging

from diffusion_models.logging.utils import build_run_logger, RunMetadataLogger

class TestBuildRunLogger:
    def test_rank_zero_creates_log_file(self, tmp_path):
        log_path = tmp_path / "run.log"
        logger = build_run_logger("run1", log_path, rank=0)
        logger.info("Hello UwU")
        for h in logger.handlers:
            h.flush()
        assert log_path.exists()
        assert "Hello UwU" in log_path.read_text()

    def test_rank_zero_creaates_parent_dirs(self, tmp_path):
        log_path = tmp_path / "a" / "b" / "run.log"
        build_run_logger("run2", log_path, rank=0)
        assert log_path.parent.exists()

    def test_rank_zero_has_file_and_stream_handlers(self, tmp_path):
        logger = build_run_logger("run3", tmp_path / "run.log", rank=0)
        types = {type(h) for h in logger.handlers}
        assert logging.FileHandler in types
        assert logging.StreamHandler in types

    def test_rank_nonzero_has_no_handlers(self, tmp_path):
        logger = build_run_logger("run4", tmp_path / "run.log", rank=1)
        assert logger.handlers == []

    def test_rank_nonzero_does_not_create_log_file(self, tmp_path):
        log_path = tmp_path / "run.log"
        build_run_logger("run5", log_path, rank=1)
        assert not log_path.exists()

    def test_nonzero_rank_does_not_propagate_to_root(self, tmp_path, capsys):
        logger = build_run_logger("run6", tmp_path / "run.log", rank=1)
        logger.info("should not print anywhere")
        captured = capsys.readouterr()
        assert "should not print anywhere" not in captured.out
        assert "should not print anywhere" not in captured.err

    def test_repeated_calls_do_not_stack_handlers(self, tmp_path):
        log_path = tmp_path / "run.log"
        build_run_logger("run7", log_path, rank=0)
        logger = build_run_logger("run7", log_path, rank=0)
        logger.info("only once")
        for h in logger.handlers:
            h.flush()
        content = log_path.read_text()
        assert content.count("only once") == 1

    def test_log_lines_include_level_and_timestamp_in_file(self, tmp_path):
        log_path = tmp_path / "run.log"
        logger = build_run_logger("run8", log_path, rank=0)
        logger.warning("careful")
        for h in logger.handlers:
            h.flush()
        content = log_path.read_text()
        assert "[WARNING]" in content
        assert "careful" in content