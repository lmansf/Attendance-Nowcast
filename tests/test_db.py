import pandas as pd
import pytest

from nowcast import db


@pytest.fixture
def engine(tmp_path):
    e = db.get_engine(f"sqlite:///{tmp_path / 'test.db'}")
    db.create_schema(e)
    return e


def _daily(source, value):
    return pd.DataFrame({"park_id": ["P1"], "visit_date": ["2024-01-01"],
                         "attendance": [value], "source": [source]})


def test_schema_is_rerunnable(engine):
    db.create_schema(engine)


def test_roundtrip_and_source_filter(engine):
    db.replace_rows(engine, "daily_attendance", _daily("simulation", 100))
    db.replace_rows(engine, "daily_attendance", _daily("production", 200))
    assert db.load_daily(engine, "P1", "simulation")["attendance"].tolist() == [100]
    assert db.load_daily(engine, "P1", "production")["attendance"].tolist() == [200]


def test_reloading_simulation_never_touches_production(engine):
    db.replace_rows(engine, "daily_attendance", _daily("production", 200))
    db.replace_rows(engine, "daily_attendance", _daily("simulation", 100))
    db.replace_rows(engine, "daily_attendance", _daily("simulation", 150))
    assert db.load_daily(engine, "P1", "simulation")["attendance"].tolist() == [150]
    assert db.load_daily(engine, "P1", "production")["attendance"].tolist() == [200]


def test_bad_source_rejected(engine):
    with pytest.raises(ValueError):
        db.replace_rows(engine, "daily_attendance", _daily("sim", 1))
    with pytest.raises(ValueError):
        db.load_daily(engine, "P1", "both")
    with pytest.raises(ValueError):
        db.replace_rows(engine, "daily_attendance", _daily("simulation", 1).drop(columns="source"))


def test_simulated_tables_all_flagged(engine):
    pid = db.ensure_simulated_data(engine)
    for table in db.TABLES:
        with engine.connect() as conn:
            sources = pd.read_sql(f"SELECT DISTINCT source FROM {table}", conn)["source"].tolist()
        assert sources == ["simulation"], table
    daily = db.load_daily(engine, pid, "simulation")
    hourly = db.load_hourly(engine, pid, "simulation")
    last = hourly.groupby("visit_date")["cumulative_entries"].last()
    assert (last.to_numpy() == daily.set_index("visit_date")["attendance"].to_numpy()).all()
