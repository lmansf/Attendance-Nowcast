-- Attendance Nowcast schema.
--
-- Portable SQL: runs on SQLite (the local default), PostgreSQL and SQL Server.
-- Every table has a `source` column, either 'simulation' or 'production', and
-- it is part of every primary key, so simulated and real rows can sit side by
-- side without ever overwriting each other. Loaders always filter on it.
--
-- Dates are local park dates; `hour` is the local clock hour a reading was
-- taken at (hour ending): hour = 10 means "entries from opening through 10:00".

CREATE TABLE IF NOT EXISTS parks (
    park_id     VARCHAR(64)  NOT NULL,
    name        VARCHAR(200) NOT NULL,
    timezone    VARCHAR(64)  NOT NULL,
    open_hour   INTEGER      NOT NULL,
    close_hour  INTEGER      NOT NULL,
    source      VARCHAR(16)  NOT NULL CHECK (source IN ('simulation', 'production')),
    PRIMARY KEY (park_id, source)
);

-- Final attendance for each operating day (the quantity we nowcast).
CREATE TABLE IF NOT EXISTS daily_attendance (
    park_id     VARCHAR(64)  NOT NULL,
    visit_date  DATE         NOT NULL,
    attendance  INTEGER      NOT NULL,
    source      VARCHAR(16)  NOT NULL CHECK (source IN ('simulation', 'production')),
    PRIMARY KEY (park_id, visit_date, source)
);

-- Cumulative gate entries at the end of each clock hour.
CREATE TABLE IF NOT EXISTS hourly_entries (
    park_id             VARCHAR(64) NOT NULL,
    visit_date          DATE        NOT NULL,
    hour                INTEGER     NOT NULL,
    cumulative_entries  INTEGER     NOT NULL,
    source              VARCHAR(16) NOT NULL CHECK (source IN ('simulation', 'production')),
    PRIMARY KEY (park_id, visit_date, hour, source)
);

-- Daily weather, same feature family as the Attendance-Projections model.
CREATE TABLE IF NOT EXISTS daily_weather (
    park_id       VARCHAR(64) NOT NULL,
    visit_date    DATE        NOT NULL,
    temp_max_c    REAL,
    temp_min_c    REAL,
    humidity_pct  REAL,
    precip_mm     REAL,
    source        VARCHAR(16) NOT NULL CHECK (source IN ('simulation', 'production')),
    PRIMARY KEY (park_id, visit_date, source)
);

-- Hidden "true" drivers of each simulated day (planned demand, crowd pace,
-- storm hour). Only simulated rows exist here: production has no ground truth
-- for these, which is exactly why the filter has to infer them.
CREATE TABLE IF NOT EXISTS simulation_log (
    park_id         VARCHAR(64) NOT NULL,
    visit_date      DATE        NOT NULL,
    planned_demand  REAL,
    pace            REAL,
    storm_hour      INTEGER,
    source          VARCHAR(16) NOT NULL CHECK (source = 'simulation'),
    PRIMARY KEY (park_id, visit_date, source)
);
