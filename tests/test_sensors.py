import pytest
import pandas as pd
import numpy as np

# ── column names (col7 dropped as unused) ──────────────────────────
COLS = ["timestamp", "Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
FORCE_COLS  = ["Fx", "Fy", "Fz"]
TORQUE_COLS = ["Tx", "Ty", "Tz"]

@pytest.fixture(scope="session")
def df():
    raw = pd.read_csv("data/windtunnel.csv", header=None)
    raw = raw.drop(columns=[6])          # drop unused col7
    raw.columns = COLS
    return raw


# ── TEST 1: no missing values ───────────────────────────────────────
def test_no_missing_values(df):
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"


# ── TEST 2: sample rate is 100 Hz (steps of ~0.01s) ────────────────
def test_sample_rate(df):
    diffs = df["timestamp"].diff().dropna()
    avg_step = diffs.mean()
    assert abs(avg_step - 0.01) < 0.001, f"Expected 0.01s steps, got {avg_step:.4f}"


# ── TEST 3: no flatlines (sensor didn't freeze) ─────────────────────
@pytest.mark.parametrize("col", FORCE_COLS + TORQUE_COLS)
def test_no_flatline(df, col):
    consecutive_same = (df[col].diff() == 0).sum()
    assert consecutive_same < len(df) * 0.5, f"{col} appears to be flatlining"


# ── TEST 4: force values within 6-axis load cell physical limits ────
@pytest.mark.parametrize("col", FORCE_COLS)
def test_force_range(df, col):
    assert df[col].between(-200, 200).all(), f"{col} has values outside ±200 N"


# ── TEST 5: torque values within physical limits ────────────────────
@pytest.mark.parametrize("col", TORQUE_COLS)
def test_torque_range(df, col):
    assert df[col].between(-50, 50).all(), f"{col} has values outside ±50 Nm"


# ── TEST 6: noise check via rolling std ────────────────────────────
@pytest.mark.parametrize("col", FORCE_COLS)
def test_noise_level(df, col):
    rolling_std = df[col].rolling(window=50).std().dropna()
    assert rolling_std.max() < 5.0, f"{col} has excessive noise spikes (rolling std > 5)"

def test_example():
    assert 2 + 2 == 4