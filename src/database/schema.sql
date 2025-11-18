-- TimescaleDB Schema for QuantCLI
-- Trading system database schema with time-series optimizations

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;

-- ============================================================================
-- MARKET DATA TABLES
-- ============================================================================

-- Daily OHLCV data
CREATE TABLE IF NOT EXISTS market_data_daily (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 6) NOT NULL,
    high NUMERIC(20, 6) NOT NULL,
    low NUMERIC(20, 6) NOT NULL,
    close NUMERIC(20, 6) NOT NULL,
    volume BIGINT NOT NULL,
    adjusted_close NUMERIC(20, 6),
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp)
);

-- Convert to hypertable (TimescaleDB time-series table)
SELECT create_hypertable('market_data_daily', 'timestamp', if_not_exists => TRUE);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_market_data_daily_symbol
    ON market_data_daily (symbol, timestamp DESC);

-- Intraday OHLCV data (1-minute bars)
CREATE TABLE IF NOT EXISTS market_data_intraday (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    open NUMERIC(20, 6) NOT NULL,
    high NUMERIC(20, 6) NOT NULL,
    low NUMERIC(20, 6) NOT NULL,
    close NUMERIC(20, 6) NOT NULL,
    volume BIGINT NOT NULL,
    data_source VARCHAR(50),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp)
);

SELECT create_hypertable('market_data_intraday', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_market_data_intraday_symbol
    ON market_data_intraday (symbol, timestamp DESC);

-- ============================================================================
-- FEATURES TABLES
-- ============================================================================

-- Engineered features
CREATE TABLE IF NOT EXISTS features (
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    value NUMERIC(20, 6) NOT NULL,
    version VARCHAR(20) DEFAULT '1.0',
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (symbol, timestamp, feature_name, version)
);

SELECT create_hypertable('features', 'timestamp', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_features_symbol_name
    ON features (symbol, feature_name, timestamp DESC);

-- ============================================================================
-- SIGNALS TABLES
-- ============================================================================

-- Trading signals
CREATE TABLE IF NOT EXISTS signals (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    signal_type VARCHAR(10) NOT NULL, -- BUY, SELL, HOLD
    strength NUMERIC(5, 4) NOT NULL CHECK (strength >= 0 AND strength <= 1),
    confidence NUMERIC(5, 4) NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    model_name VARCHAR(100),
    model_version VARCHAR(20),
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_symbol_timestamp
    ON signals (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_timestamp
    ON signals (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_signals_metadata
    ON signals USING GIN (metadata);

-- ============================================================================
-- TRADES TABLES
-- ============================================================================

-- Executed trades
CREATE TABLE IF NOT EXISTS trades (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    side VARCHAR(10) NOT NULL, -- BUY, SELL
    quantity NUMERIC(20, 6) NOT NULL,
    price NUMERIC(20, 6) NOT NULL,
    commission NUMERIC(20, 6) DEFAULT 0,
    slippage NUMERIC(20, 6) DEFAULT 0,
    order_id VARCHAR(100),
    strategy_name VARCHAR(100),
    status VARCHAR(20) DEFAULT 'FILLED', -- PENDING, FILLED, CANCELLED, REJECTED
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_trades_symbol_timestamp
    ON trades (symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp
    ON trades (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_trades_order_id
    ON trades (order_id);
CREATE INDEX IF NOT EXISTS idx_trades_status
    ON trades (status, timestamp DESC);

-- ============================================================================
-- POSITIONS TABLES
-- ============================================================================

-- Current positions
CREATE TABLE IF NOT EXISTS positions (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL UNIQUE,
    quantity NUMERIC(20, 6) NOT NULL,
    avg_entry_price NUMERIC(20, 6) NOT NULL,
    current_price NUMERIC(20, 6),
    unrealized_pnl NUMERIC(20, 6),
    realized_pnl NUMERIC(20, 6) DEFAULT 0,
    opened_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_positions_symbol
    ON positions (symbol);

-- Position history
CREATE TABLE IF NOT EXISTS position_history (
    id SERIAL PRIMARY KEY,
    symbol VARCHAR(20) NOT NULL,
    quantity NUMERIC(20, 6) NOT NULL,
    avg_entry_price NUMERIC(20, 6) NOT NULL,
    exit_price NUMERIC(20, 6),
    realized_pnl NUMERIC(20, 6),
    opened_at TIMESTAMPTZ NOT NULL,
    closed_at TIMESTAMPTZ,
    holding_period_days NUMERIC(10, 2),
    return_pct NUMERIC(10, 4),
    metadata JSONB
);

CREATE INDEX IF NOT EXISTS idx_position_history_symbol_closed
    ON position_history (symbol, closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_position_history_closed
    ON position_history (closed_at DESC);

-- ============================================================================
-- MODELS TABLES
-- ============================================================================

-- Model metadata and performance
CREATE TABLE IF NOT EXISTS models (
    id SERIAL PRIMARY KEY,
    model_name VARCHAR(100) NOT NULL,
    version VARCHAR(20) NOT NULL,
    model_type VARCHAR(50) NOT NULL, -- XGBOOST, LIGHTGBM, etc.
    task VARCHAR(20) NOT NULL, -- classification, regression
    metrics JSONB NOT NULL,
    config JSONB,
    file_path VARCHAR(500),
    status VARCHAR(20) DEFAULT 'STAGING', -- STAGING, PRODUCTION, ARCHIVED
    trained_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    UNIQUE(model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_models_status
    ON models (status, trained_at DESC);
CREATE INDEX IF NOT EXISTS idx_models_name_version
    ON models (model_name, version);

-- ============================================================================
-- PERFORMANCE TABLES
-- ============================================================================

-- Daily performance metrics
CREATE TABLE IF NOT EXISTS performance_daily (
    date DATE NOT NULL PRIMARY KEY,
    total_pnl NUMERIC(20, 6) NOT NULL,
    unrealized_pnl NUMERIC(20, 6),
    realized_pnl NUMERIC(20, 6),
    equity NUMERIC(20, 6) NOT NULL,
    n_trades INTEGER DEFAULT 0,
    n_winning_trades INTEGER DEFAULT 0,
    avg_trade_return NUMERIC(10, 4),
    sharpe_ratio NUMERIC(10, 4),
    max_drawdown NUMERIC(10, 4),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_performance_daily_date
    ON performance_daily (date DESC);

-- ============================================================================
-- COMPRESSION & RETENTION POLICIES
-- ============================================================================

-- Compress old data (older than 7 days)
SELECT add_compression_policy('market_data_daily', INTERVAL '7 days', if_not_exists => TRUE);
SELECT add_compression_policy('market_data_intraday', INTERVAL '2 days', if_not_exists => TRUE);
SELECT add_compression_policy('features', INTERVAL '7 days', if_not_exists => TRUE);

-- Retention policy: Drop data older than 3 years
SELECT add_retention_policy('market_data_intraday', INTERVAL '90 days', if_not_exists => TRUE);

-- ============================================================================
-- CONTINUOUS AGGREGATES (Materialized Views)
-- ============================================================================

-- Hourly OHLCV aggregates from minute data
CREATE MATERIALIZED VIEW IF NOT EXISTS market_data_hourly
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 hour', timestamp) AS hour,
    first(open, timestamp) AS open,
    max(high) AS high,
    min(low) AS low,
    last(close, timestamp) AS close,
    sum(volume) AS volume
FROM market_data_intraday
GROUP BY symbol, hour
WITH NO DATA;

-- Refresh policy for continuous aggregate
SELECT add_continuous_aggregate_policy('market_data_hourly',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE);

-- Daily aggregate statistics
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_statistics
WITH (timescaledb.continuous) AS
SELECT
    symbol,
    time_bucket('1 day', timestamp) AS day,
    avg(close) AS avg_price,
    stddev(close) AS price_std,
    sum(volume) AS total_volume,
    max(high) AS day_high,
    min(low) AS day_low
FROM market_data_daily
GROUP BY symbol, day
WITH NO DATA;

SELECT add_continuous_aggregate_policy('daily_statistics',
    start_offset => INTERVAL '3 days',
    end_offset => INTERVAL '1 day',
    schedule_interval => INTERVAL '1 day',
    if_not_exists => TRUE);

-- ============================================================================
-- HELPER FUNCTIONS
-- ============================================================================

-- Function to calculate returns
CREATE OR REPLACE FUNCTION calculate_returns(
    p_symbol VARCHAR(20),
    p_start_date TIMESTAMPTZ,
    p_end_date TIMESTAMPTZ
)
RETURNS TABLE (
    timestamp TIMESTAMPTZ,
    close_price NUMERIC,
    daily_return NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        t.timestamp,
        t.close,
        (t.close - LAG(t.close) OVER (ORDER BY t.timestamp)) / LAG(t.close) OVER (ORDER BY t.timestamp) AS daily_return
    FROM market_data_daily t
    WHERE t.symbol = p_symbol
      AND t.timestamp BETWEEN p_start_date AND p_end_date
    ORDER BY t.timestamp;
END;
$$ LANGUAGE plpgsql;

-- Function to get latest price
CREATE OR REPLACE FUNCTION get_latest_price(p_symbol VARCHAR(20))
RETURNS NUMERIC AS $$
DECLARE
    latest_price NUMERIC;
BEGIN
    SELECT close INTO latest_price
    FROM market_data_daily
    WHERE symbol = p_symbol
    ORDER BY timestamp DESC
    LIMIT 1;

    RETURN latest_price;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Active positions with current prices
CREATE OR REPLACE VIEW active_positions_view AS
SELECT
    p.symbol,
    p.quantity,
    p.avg_entry_price,
    p.current_price,
    p.unrealized_pnl,
    p.realized_pnl,
    p.opened_at,
    p.updated_at,
    (p.current_price - p.avg_entry_price) / p.avg_entry_price * 100 AS return_pct,
    EXTRACT(EPOCH FROM (NOW() - p.opened_at)) / 86400 AS holding_days
FROM positions p
WHERE p.quantity != 0;

-- Recent trades summary
CREATE OR REPLACE VIEW recent_trades_view AS
SELECT
    symbol,
    COUNT(*) AS n_trades,
    SUM(CASE WHEN side = 'BUY' THEN quantity ELSE 0 END) AS total_bought,
    SUM(CASE WHEN side = 'SELL' THEN quantity ELSE 0 END) AS total_sold,
    AVG(price) AS avg_price,
    MAX(timestamp) AS last_trade_time
FROM trades
WHERE timestamp > NOW() - INTERVAL '30 days'
GROUP BY symbol;

-- Model performance comparison
CREATE OR REPLACE VIEW model_performance_view AS
SELECT
    model_name,
    version,
    model_type,
    status,
    metrics->>'accuracy' AS accuracy,
    metrics->>'sharpe_ratio' AS sharpe_ratio,
    trained_at,
    created_at
FROM models
ORDER BY trained_at DESC;

COMMENT ON DATABASE quantcli IS 'QuantCLI - Algorithmic Trading System Database';
