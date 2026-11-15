"""Shared test fixtures and utilities for surgical_lab tests."""

import os

import torch
import warp as wp


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


# Silence warp during tests.
wp.config.quiet = True
