"""
Tests for metadata.py: CSV run-metadata logging
"""
import csv
import json
import pytest

from diffusion_models.config.config import (
    ComponentConfig,
    ScoreModelConfig,
    TrainerConfig,
    RunConfig,
    ExperimentConfig
)
from diffusion_models.logging.utils import (
    RunMetadataLogger,
    flatten_experiment_config,
    csv_fieldnames_for_experiment_config,
    LogStatus
)

@pytest.fixture
def config():
    return ExperimentConfig(
        network=ComponentConfig(name="unet", params={"in_channels": 3}),
        schedule=ComponentConfig(name="geometric", params={"sigma_min": 1.0}),
        trainer=TrainerConfig(file_path="run001"),
        run=RunConfig(N_epochs=10),
        model=ScoreModelConfig(decay_rate=0.5, device="cuda"),
    )
 
 
def _read_rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class TestFieldNames:
    def test_includes_flat_and_nested_columns(self):
        names = csv_fieldnames_for_experiment_config()
        assert "network.name" in names
        assert "network.params" in names
        assert "model.decay_rate" in names
        assert "trainer.file_path" in names
        assert "run.N_epochs" in names

    def test_does_not_include_extra_run_columns(self):
        names = csv_fieldnames_for_experiment_config()
        assert "run_id" not in names
        assert "status" not in names

class TestFlatten:
    def test_flat_scalar_fields(self, config: ExperimentConfig):
        row = flatten_experiment_config(config)
        assert row["network.name"] == "unet"
        assert row["trainer.file_path"] == "run001"
        assert row["run.N_epochs"] == "10"
        assert row["model.decay_rate"] == "0.5"

    def test_params_dict_is_json_encoded(self, config: ExperimentConfig):
        row = flatten_experiment_config(config)
        assert isinstance(row["network.params"], str)
        assert json.loads(row["network.params"]) == {"in_channels" : 3}

    def test_none_section_leaves_columns_blank(self, config: ExperimentConfig):
        assert config.lr_scheduler is None
        row = flatten_experiment_config(config)
        assert row["lr_scheduler.name"] == ""
        assert row["lr_scheduler.params"] == ""
 
    def test_none_scalar_leaves_column_blank(self):
        config = ExperimentConfig(
            network=ComponentConfig(name="unet"),
            schedule=ComponentConfig(name="geometric"),
            trainer=TrainerConfig(file_path="run"),
            run=RunConfig(N_epochs=1),
            model=ScoreModelConfig(device=None),
        )
        row = flatten_experiment_config(config)
        assert row["model.device"] == ""


 
class TestRunMetadataLogger:
    def test_creates_file_with_header_on_first_call(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        logger = RunMetadataLogger(path)
        logger.log_run(config, run_id="run001")
 
        assert path.exists()
        rows = _read_rows(path)
        assert len(rows) == 1
        assert rows[0]["run_id"] == "run001"
 
    def test_creates_parent_directory(self, tmp_path, config):
        path = tmp_path / "a" / "b" / "metadata.csv"
        RunMetadataLogger(path).log_run(config, run_id="run001")
        assert path.exists()
 
    def test_appends_without_duplicating_header(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        logger = RunMetadataLogger(path)
        logger.log_run(config, run_id="run001", status=LogStatus.STARTED)
        logger.log_run(config, run_id="run001", status=LogStatus.COMPLETED, final_epoch=9, final_loss=0.01)
 
        rows = _read_rows(path)
        assert len(rows) == 2
        assert rows[0]["status"] == LogStatus.STARTED
        assert rows[1]["status"] == LogStatus.COMPLETED
        # only one header line: file has exactly len(rows)+1 lines
        assert path.read_text().count("\n") == 3
 
    def test_default_status_is_started_with_blank_outcome(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        RunMetadataLogger(path).log_run(config, run_id="run001")
        row = _read_rows(path)[0]
        assert row["status"] == LogStatus.STARTED
        assert row["final_epoch"] == ""
        assert row["final_loss"] == ""
 
    def test_end_row_records_final_epoch_and_loss(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        RunMetadataLogger(path).log_run(
            config, run_id="run001", status=LogStatus.COMPLETED, final_epoch=42, final_loss=0.0123
        )
        row = _read_rows(path)[0]
        assert row["final_epoch"] == "42"
        assert row["final_loss"] == "0.0123"
 
    def test_interrupted_status_is_recorded_verbatim(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        RunMetadataLogger(path).log_run(
            config, run_id="run001", status=LogStatus.INTERRUPTED, final_epoch=7, final_loss=None
        )
        row = _read_rows(path)[0]
        assert row["status"] == LogStatus.INTERRUPTED
        assert row["final_epoch"] == "7"
        assert row["final_loss"] == ""
 
    def test_log_file_column_stringified(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        RunMetadataLogger(path).log_run(config, run_id="run001", log_file=tmp_path / "run.log")
        row = _read_rows(path)[0]
        assert row["log_file"] == str(tmp_path / "run.log")
 
    def test_row_is_self_contained_config_and_outcome(self, tmp_path, config):
        # Each row carries the full config, which is enough to know both what a run used and how it ended.
        path = tmp_path / "metadata.csv"
        RunMetadataLogger(path).log_run(
            config, run_id="run001", status=LogStatus.COMPLETED, final_epoch=9, final_loss=0.01
        )
        row = _read_rows(path)[0]
        assert row["network.name"] == "unet"
        assert row["status"] == LogStatus.COMPLETED
 
    def test_different_runs_get_different_run_ids(self, tmp_path, config):
        path = tmp_path / "metadata.csv"
        logger = RunMetadataLogger(path)
        logger.log_run(config, run_id="run_a")
        logger.log_run(config, run_id="run_b")
        rows = _read_rows(path)
        assert {r["run_id"] for r in rows} == {"run_a", "run_b"}
