"""Read and write the nowcast tables in any SQL database.

The connection comes from a SQLAlchemy URL:

* default: a local SQLite file, ``data/nowcast.db`` (no server needed)
* production: set the ``NOWCAST_DB_URL`` environment variable, e.g.
  ``postgresql+psycopg://user:pass@host/dbname``

Every table carries a ``source`` column, ``"simulation"`` or ``"production"``.
Every loader takes ``source`` as a required argument, so a notebook can never
mix simulated and real rows by accident.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

SOURCES = ("simulation", "production")
TABLES = ("parks", "daily_attendance", "hourly_entries", "daily_weather", "simulation_log")

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URL = f"sqlite:///{REPO_ROOT / 'data' / 'nowcast.db'}"
SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def get_engine(url: str | None = None) -> Engine:
    """Connect to ``url``, else ``$NOWCAST_DB_URL``, else the local SQLite file."""
    url = url or os.environ.get("NOWCAST_DB_URL") or DEFAULT_URL
    if url.startswith("sqlite:///"):
        Path(url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    return create_engine(url)


def create_schema(engine: Engine) -> None:
    """Create any missing tables from ``schema.sql`` (safe to re-run)."""
    # Remove comment lines first (they may contain ';'), then split statements.
    lines = [l for l in SCHEMA_FILE.read_text().splitlines() if not l.strip().startswith("--")]
    statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]
    with engine.begin() as conn:
        for statement in statements:
            conn.execute(text(statement))


def _check_source(source: str) -> None:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}")


def replace_rows(engine: Engine, table: str, df: pd.DataFrame) -> int:
    """Write ``df`` to ``table``, first deleting rows with the same park_id and source.

    Only the (park_id, source) pairs present in ``df`` are replaced, so
    reloading simulated data never touches production rows and vice versa.
    """
    if table not in TABLES:
        raise ValueError(f"unknown table {table!r}")
    if "source" not in df.columns:
        raise ValueError("every row needs a 'source' column ('simulation' or 'production')")
    for s in df["source"].unique():
        _check_source(s)

    pairs = df[["park_id", "source"]].drop_duplicates().itertuples(index=False)
    with engine.begin() as conn:
        for park_id, source in pairs:
            conn.execute(text(f"DELETE FROM {table} WHERE park_id = :p AND source = :s"),
                         {"p": park_id, "s": source})
        df.to_sql(table, conn, if_exists="append", index=False)
    return len(df)


def _load(engine: Engine, table: str, park_id: str, source: str, order_by: str) -> pd.DataFrame:
    _check_source(source)
    query = text(f"SELECT * FROM {table} WHERE park_id = :p AND source = :s ORDER BY {order_by}")
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, params={"p": park_id, "s": source})
    if "visit_date" in df.columns:
        df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df


def load_daily(engine: Engine, park_id: str, source: str) -> pd.DataFrame:
    """Final daily attendance: visit_date, attendance."""
    return _load(engine, "daily_attendance", park_id, source, "visit_date")


def load_hourly(engine: Engine, park_id: str, source: str) -> pd.DataFrame:
    """Cumulative gate counts: visit_date, hour, cumulative_entries."""
    return _load(engine, "hourly_entries", park_id, source, "visit_date, hour")


def load_weather(engine: Engine, park_id: str, source: str) -> pd.DataFrame:
    """Daily weather features."""
    return _load(engine, "daily_weather", park_id, source, "visit_date")


def load_simulation_log(engine: Engine, park_id: str) -> pd.DataFrame:
    """Hidden drivers of simulated days (only exists for source='simulation')."""
    return _load(engine, "simulation_log", park_id, "simulation", "visit_date")


def ensure_simulated_data(engine: Engine, rebuild: bool = False, seed: int = 42) -> str:
    """Create the schema and load the SIMULATED Tampa park if it is not there yet.

    Returns the simulated park_id. Production rows are never touched.
    """
    from nowcast.simulate import PARK, simulate_park

    create_schema(engine)
    with engine.connect() as conn:
        n = conn.execute(text("SELECT COUNT(*) FROM daily_attendance WHERE park_id = :p "
                              "AND source = 'simulation'"), {"p": PARK["park_id"]}).scalar()
    if rebuild or n == 0:
        for table, df in simulate_park(seed=seed).items():
            replace_rows(engine, table, df)
    return PARK["park_id"]
