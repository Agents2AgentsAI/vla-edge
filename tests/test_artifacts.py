"""Prompt-length limits and serving defaults for prebuilt engine sets.

These run without tensorrt installed: the logic under test decides what gets
refused before an engine ever executes, so it must be testable everywhere.
"""

from __future__ import annotations

import json

import pytest

from vla_edge.backends.tensorrt.artifacts import (
    ArtifactMismatch,
    Environment,
    check_compatible,
    effective_token_limit,
    exact_device_match,
    load_serving_config,
)


class TestEffectiveTokenLimit:
    def test_dynamic_set_limit_is_the_profile_bound(self):
        assert effective_token_limit(1024, 1) == 1024

    def test_fixed_bracket_limit_is_the_bracket_not_the_profile(self):
        """The champion sets: profile ends at 1024, bracket is 704.

        A 705-token prompt pads to 1408 and cannot execute, so the number
        quoted to the user must be 704. Quoting 1024 sends them counting
        the wrong budget.
        """
        assert effective_token_limit(1024, 704) == 704

    def test_bracket_equal_to_profile(self):
        assert effective_token_limit(704, 704) == 704

    def test_multiple_brackets_fit(self):
        assert effective_token_limit(1024, 256) == 1024
        assert effective_token_limit(1000, 256) == 768


class TestServingConfig:
    def test_absent_file_means_no_defaults(self, tmp_path):
        assert load_serving_config(tmp_path) == {}

    def test_declared_pad_multiple_is_returned(self, tmp_path):
        (tmp_path / "serving.json").write_text(json.dumps({"pad_multiple": 704}))
        assert load_serving_config(tmp_path)["pad_multiple"] == 704

    def test_non_object_config_is_rejected(self, tmp_path):
        (tmp_path / "serving.json").write_text("[704]")
        with pytest.raises(ArtifactMismatch):
            load_serving_config(tmp_path)


class TestExactDeviceCompatibility:
    @staticmethod
    def _manifest(tmp_path, **overrides):
        requires = {
            "compute_capability": "sm_110a",
            "cuda_device_name": "NVIDIA Thor",
            "multiprocessor_count": 20,
            "board_model": "NVIDIA Jetson AGX Thor Developer Kit",
            "arch": "aarch64",
            **overrides,
        }
        (tmp_path / "MANIFEST.json").write_text(json.dumps({"requires": requires}))

    @staticmethod
    def _runtime():
        return Environment(
            tensorrt=None,
            arch="aarch64",
            compute_capability="sm_110",
            cuda_device_name="NVIDIA Thor",
            multiprocessor_count=20,
            board_model="NVIDIA Jetson AGX Thor Developer Kit",
        )

    def test_exact_match_requires_and_compares_model_fields(
        self, tmp_path, monkeypatch
    ):
        self._manifest(tmp_path)
        monkeypatch.setattr(
            Environment, "detect", classmethod(lambda cls: self._runtime())
        )

        assert check_compatible(tmp_path) == []
        assert exact_device_match(tmp_path)

    def test_different_sm_count_is_rejected(self, tmp_path, monkeypatch):
        self._manifest(tmp_path, multiprocessor_count=24)
        monkeypatch.setattr(
            Environment, "detect", classmethod(lambda cls: self._runtime())
        )

        with pytest.raises(ArtifactMismatch, match="20 SMs"):
            check_compatible(tmp_path)
        assert not exact_device_match(tmp_path)

    def test_old_manifest_does_not_authorize_warning_filter(
        self, tmp_path, monkeypatch
    ):
        (tmp_path / "MANIFEST.json").write_text(
            json.dumps({"requires": {"compute_capability": "sm_110a"}})
        )
        monkeypatch.setattr(
            Environment, "detect", classmethod(lambda cls: self._runtime())
        )

        assert not exact_device_match(tmp_path)
