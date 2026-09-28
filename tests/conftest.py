"""Shared fixtures: tiny random Qwen3 + Mimi models (built once per test session).

Needs the Qwen3 tokenizer (downloaded from the Hugging Face Hub on first use).
"""

from __future__ import annotations

import os

import pytest
import torch


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory):
    from s2s.prep.smoke import tiny_models as build

    root = tmp_path_factory.mktemp("tiny")
    try:
        qwen_dir, mimi_dir = build(str(root), os.environ.get("S2S_TEST_TOKENIZER", "Qwen/Qwen3-0.6B"))
    except OSError as e:  # offline and not cached
        pytest.skip(f"Qwen3 tokenizer unavailable: {e}")
    return {"root": str(root), "qwen": qwen_dir, "mimi": mimi_dir}


@pytest.fixture(scope="session")
def cpu():
    return torch.device("cpu")
