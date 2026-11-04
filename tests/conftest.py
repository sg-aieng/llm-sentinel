"""Disable ML model loading for unit tests.

Tests inject fakes for the classifier and PII detector; loading the real
models (and downloading them on first run) would only slow the suite.
Must run before app.config is imported, hence env vars set at conftest
import time.
"""
import os

os.environ.setdefault("SENTINEL_DISABLE_CLASSIFIER", "true")
os.environ.setdefault("SENTINEL_DISABLE_PII", "true")
os.environ.setdefault("SENTINEL_DISABLE_SAFETY", "true")
