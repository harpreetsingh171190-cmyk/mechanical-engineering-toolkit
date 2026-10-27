import numpy as np
import pandas as pd
import pytest
from mechanical_ds.analytics import eda_summary, detect_anomalies, moving_average_forecast, forecast_metrics, simulate_mass_spring, optimize_beam_depth
from mechanical_ds.storage import HistoryStore

def test_eda_summary():
    r=eda_summary(pd.DataFrame({"temp":[1.,2.,3.],"missing":[1.,None,3.]}))
    assert r["rows"]==3 and r["missing_by_column"]["missing"]==1

def test_anomaly_spike():
    flags=detect_anomalies([10,10,10,10,10,10,50,10,10,10,10])
    assert flags[6]

def test_forecast_and_metrics():
    assert moving_average_forecast([1,2,3],2,2).tolist()==[2.5,2.5]
    assert forecast_metrics([1,2],[2,2])["mae"]==pytest.approx(.5)

def test_simulation_initial_conditions():
    r=simulate_mass_spring(1,4,0,.1,0,duration_s=1,samples=51)
    assert r["displacement_m"][0]==pytest.approx(.1)
    assert len(r["time_s"])==51

def test_optimizer_meets_stress():
    r=optimize_beam_depth(1000,2,.05,150e6)
    assert r["calculated_stress_pa"] <= 150e6*(1+1e-8)

def test_history_store(tmp_path):
    db=HistoryStore(tmp_path/"history.db"); i=db.add("test",{"x":2})
    assert db.list()[0]["id"]==i and db.list()[0]["payload"]["x"]==2

def test_invalid_forecast():
    with pytest.raises(ValueError): moving_average_forecast([],1)
