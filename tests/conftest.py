import pytest
import pandas as pd

@pytest.fixture(scope="session")
def sensor_data():
    df = pd.read_csv("data/windtunnel.csv")
    return df