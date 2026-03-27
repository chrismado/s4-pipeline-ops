"""Tests for job progress parsing."""

from src.scheduler.progress import parse_progress


class TestProgressParser:
    def test_bracket_fraction(self):
        assert parse_progress("[50/100]") == 50.0
        assert parse_progress("[1/10]") == 10.0
        assert parse_progress("[ 99 / 100 ]") == 99.0

    def test_epoch_counter(self):
        assert parse_progress("Epoch 5/10") == 50.0
        assert parse_progress("epoch 3/20") == 15.0

    def test_step_counter(self):
        assert parse_progress("Step 500/1000") == 50.0
        assert parse_progress("Batch 75/300") == 25.0
        assert parse_progress("Frame 10/100 rendered") == 10.0

    def test_percentage(self):
        assert parse_progress("75%") == 75.0
        assert parse_progress("Progress: 50.5%") == 50.5
        assert parse_progress("100%") == 100.0

    def test_decimal_fraction(self):
        assert parse_progress("progress: 0.75") == 75.0
        assert parse_progress("Progress=0.5") == 50.0
        assert parse_progress("progress: 1.0") == 100.0

    def test_no_match(self):
        assert parse_progress("Starting training...") is None
        assert parse_progress("Loss: 0.0234") is None
        assert parse_progress("") is None

    def test_clamped_to_100(self):
        assert parse_progress("[150/100]") == 100.0
        assert parse_progress("150%") == 100.0
