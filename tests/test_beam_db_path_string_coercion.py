"""
Regression test: ``BeamMemory(db_path=<str>)`` must work.

PR #106 added ``tests/test_identity_memory.py`` whose fixtures pass a
string ``db_path`` directly:

    beam = BeamMemory(db_path=db_path)  # db_path is a tempfile string

Pre-fix, ``BeamMemory.__init__`` stored the string as-is, then
``_get_connection`` did ``path.parent.mkdir(...)`` which raised
``AttributeError: 'str' object has no attribute 'parent'`` -- breaking
every PR's CI matrix on the test_identity_memory.py setup phase.

Fix: ``BeamMemory.__init__`` coerces any non-Path input via
``Path(db_path)`` before storage. Real ``Path`` inputs stay ``Path``;
strings get coerced; ``None`` still gets ``_default_db_path()``.

Run with: pytest tests/test_beam_db_path_string_coercion.py -v
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from mnemosyne.core.beam import BeamMemory


class TestBeamMemoryDbPathCoercion:
    """``db_path`` accepts str / Path / None uniformly."""

    def test_accepts_string_db_path(self, tmp_path):
        """The PR #106 fixture pattern: pass a string path. Pre-fix this
        crashed in _get_connection with AttributeError on .parent."""
        db_path_str = str(tmp_path / "mnemosyne.db")
        # Must not raise
        beam = BeamMemory(session_id="s1", db_path=db_path_str)
        # And internal state is now a Path (so downstream consumers
        # that expect .parent etc. work)
        assert isinstance(beam.db_path, Path)
        assert beam.db_path == Path(db_path_str)
        beam.conn.close()

    def test_accepts_path_db_path(self, tmp_path):
        """Pre-fix behavior preserved for Path callers."""
        db_path = tmp_path / "mnemosyne.db"
        beam = BeamMemory(session_id="s1", db_path=db_path)
        assert isinstance(beam.db_path, Path)
        assert beam.db_path == db_path
        beam.conn.close()

    def test_accepts_none_db_path(self, tmp_path, monkeypatch):
        """None falls back to _default_db_path() (a Path)."""
        monkeypatch.setenv("MNEMOSYNE_DATA_DIR", str(tmp_path))
        beam = BeamMemory(session_id="s1", db_path=None)
        assert isinstance(beam.db_path, Path)
        beam.conn.close()

    def test_get_connection_accepts_string_after_coercion(self, tmp_path):
        """End-to-end smoke: the failing operation pre-fix was
        ``self.db_path.parent.mkdir(...)`` inside ``_get_connection``.
        After the coercion the call site doesn't change but ``self.db_path``
        is now always a Path, so ``.parent`` works."""
        db_path_str = str(tmp_path / "subdir" / "mnemosyne.db")
        # Note: subdir doesn't exist yet -- mkdir(parents=True) creates it.
        # Pre-fix this raised AttributeError on the path-as-string.
        beam = BeamMemory(session_id="s1", db_path=db_path_str)
        assert (tmp_path / "subdir").is_dir()
        beam.conn.close()
