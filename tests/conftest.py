"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def data_dir() -> str:
    return "tests/data"
