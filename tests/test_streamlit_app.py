"""Automated smoke test for the precomputed six-tab Streamlit app."""
from __future__ import annotations

import pathlib

from streamlit.testing.v1 import AppTest


ROOT = pathlib.Path(__file__).resolve().parent.parent


def test_streamlit_app_renders_without_exceptions():
    app = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=30)
    app.run()

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == [
        "Compare Funds",
        "Fund Fact Sheets",
        "Build an Allocation",
        "News Pulse",
        "Downside Innovation",
        "Method Notes",
    ]
