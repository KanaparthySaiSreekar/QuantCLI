-- TimescaleDB initialization for QuantCLI
-- Creates database schema, hypertables, and indexes for optimal performance

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Create MLflow database for experiment tracking
CREATE DATABASE mlflow;

-- Market data tables
CREATE TABLE IF NOT EXISTS market_data (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    open NUMERIC(12, 4),
    high NUMERIC(12, 4),
    low NUMERIC(12, 4),
    close NUMERIC(12, 4),
    volume BIGINT,
    adj_close NUMERIC(12, 4),
    data_source VARCHAR(20),
    PRIMARY KEY (time, symbol)
);

-- Convert to hypertable for time-series optimization
SELECT create_hypertable('market_data', 'time', if_not_exists => TRUE);

-- Create indexes for fast queries
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data (symbol, time DESC);

-- Add compression policy (compress data older than 7 days)
ALTER TABLE market_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'symbol'
);

SELECT add_compression_policy('market_data', INTERVAL '7 days', if_not_exists => TRUE);

-- News sentiment table
CREATE TABLE IF NOT EXISTS news_sentiment (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    headline TEXT,
    source VARCHAR(50),
    sentiment_score NUMERIC(5, 4),
    sentiment_label VARCHAR(20),
    relevance_score NUMERIC(5, 4),
    PRIMARY KEY (time, symbol, source)
);

SELECT create_hypertable('news_sentiment', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_symbol_time ON news_sentiment (symbol, time DESC);

-- Social media sentiment table
CREATE TABLE IF NOT EXISTS social_sentiment (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    platform VARCHAR(20),
    post_id VARCHAR(100),
    text TEXT,
    sentiment_score NUMERIC(5, 4),
    upvotes INTEGER,
    comments INTEGER,
    PRIMARY KEY (time, symbol, post_id)
);

SELECT create_hypertable('social_sentiment', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_social_sentiment_symbol_time ON social_sentiment (symbol, time DESC);

-- Macroeconomic indicators table
CREATE TABLE IF NOT EXISTS macro_indicators (
    time TIMESTAMPTZ NOT NULL,
    indicator VARCHAR(50) NOT NULL,
    value NUMERIC(15, 6),
    data_source VARCHAR(20),
    PRIMARY KEY (time, indicator)
);

SELECT create_hypertable('macro_indicators', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_macro_indicators_indicator_time ON macro_indicators (indicator, time DESC);

-- Feature cache table (pre-computed features)
CREATE TABLE IF NOT EXISTS features (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    feature_name VARCHAR(100) NOT NULL,
    feature_value NUMERIC(15, 8),
    PRIMARY KEY (time, symbol, feature_name)
);

SELECT create_hypertable('features', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_features_symbol_time ON features (symbol, time DESC);

-- Model predictions table
CREATE TABLE IF NOT EXISTS predictions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    model_name VARCHAR(50) NOT NULL,
    prediction NUMERIC(10, 6),
    confidence NUMERIC(5, 4),
    metadata JSONB,
    PRIMARY KEY (time, symbol, model_name)
);

SELECT create_hypertable('predictions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_predictions_symbol_time ON predictions (symbol, time DESC);

-- Trading signals table
CREATE TABLE IF NOT EXISTS signals (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    signal_type VARCHAR(10) CHECK (signal_type IN ('buy', 'sell', 'hold')),
    quantity INTEGER,
    expected_return NUMERIC(10, 6),
    confidence NUMERIC(5, 4),
    strategy VARCHAR(50),
    regime VARCHAR(20),
    metadata JSONB,
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('signals', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_signals_symbol_time ON signals (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_signals_signal_type ON signals (signal_type);

-- Orders table (immutable audit trail)
CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR(50) NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) CHECK (side IN ('buy', 'sell')),
    quantity INTEGER,
    order_type VARCHAR(20),
    limit_price NUMERIC(12, 4),
    status VARCHAR(20),
    broker VARCHAR(20),
    account VARCHAR(50),
    metadata JSONB,
    PRIMARY KEY (order_id, time)
);

SELECT create_hypertable('orders', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_orders_symbol_time ON orders (symbol, time DESC);
CREATE INDEX IF NOT EXISTS idx_orders_status ON orders (status);

-- Executions table (filled orders)
CREATE TABLE IF NOT EXISTS executions (
    execution_id VARCHAR(50) NOT NULL,
    order_id VARCHAR(50) NOT NULL,
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    side VARCHAR(10) CHECK (side IN ('buy', 'sell')),
    quantity INTEGER,
    fill_price NUMERIC(12, 4),
    commission NUMERIC(10, 4),
    slippage NUMERIC(10, 6),
    venue VARCHAR(50),
    metadata JSONB,
    PRIMARY KEY (execution_id, time)
);

SELECT create_hypertable('executions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_executions_order_id ON executions (order_id);
CREATE INDEX IF NOT EXISTS idx_executions_symbol_time ON executions (symbol, time DESC);

-- Positions table (current holdings)
CREATE TABLE IF NOT EXISTS positions (
    time TIMESTAMPTZ NOT NULL,
    symbol VARCHAR(10) NOT NULL,
    quantity INTEGER,
    avg_cost NUMERIC(12, 4),
    current_price NUMERIC(12, 4),
    unrealized_pnl NUMERIC(15, 4),
    realized_pnl NUMERIC(15, 4),
    PRIMARY KEY (time, symbol)
);

SELECT create_hypertable('positions', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_positions_symbol_time ON positions (symbol, time DESC);

-- Portfolio performance table
CREATE TABLE IF NOT EXISTS portfolio_performance (
    time TIMESTAMPTZ NOT NULL,
    total_value NUMERIC(15, 4),
    cash NUMERIC(15, 4),
    positions_value NUMERIC(15, 4),
    daily_pnl NUMERIC(15, 4),
    daily_return NUMERIC(10, 6),
    cumulative_return NUMERIC(10, 6),
    sharpe_ratio NUMERIC(8, 4),
    max_drawdown NUMERIC(8, 4),
    win_rate NUMERIC(5, 4),
    PRIMARY KEY (time)
);

SELECT create_hypertable('portfolio_performance', 'time', if_not_exists => TRUE);

-- Risk metrics table
CREATE TABLE IF NOT EXISTS risk_metrics (
    time TIMESTAMPTZ NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    metric_value NUMERIC(15, 6),
    threshold NUMERIC(15, 6),
    status VARCHAR(20),
    PRIMARY KEY (time, metric_name)
);

SELECT create_hypertable('risk_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_risk_metrics_name_time ON risk_metrics (metric_name, time DESC);

-- System events table (kill switches, errors, alerts)
CREATE TABLE IF NOT EXISTS system_events (
    time TIMESTAMPTZ NOT NULL,
    event_type VARCHAR(50),
    severity VARCHAR(20) CHECK (severity IN ('info', 'warning', 'error', 'critical')),
    message TEXT,
    metadata JSONB,
    PRIMARY KEY (time, event_type)
);

SELECT create_hypertable('system_events', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events (severity, time DESC);

-- Performance metrics table (latency tracking)
CREATE TABLE IF NOT EXISTS performance_metrics (
    time TIMESTAMPTZ NOT NULL,
    component VARCHAR(50) NOT NULL,
    metric_name VARCHAR(50) NOT NULL,
    value NUMERIC(15, 6),
    unit VARCHAR(20),
    PRIMARY KEY (time, component, metric_name)
);

SELECT create_hypertable('performance_metrics', 'time', if_not_exists => TRUE);
CREATE INDEX IF NOT EXISTS idx_performance_metrics_component ON performance_metrics (component, time DESC);

-- Create materialized views for common queries
CREATE MATERIALIZED VIEW IF NOT EXISTS daily_performance AS
SELECT
    time_bucket('1 day', time) AS day,
    symbol,
    FIRST(open, time) AS open,
    MAX(high) AS high,
    MIN(low) AS low,
    LAST(close, time) AS close,
    SUM(volume) AS volume
FROM market_data
GROUP BY day, symbol
WITH NO DATA;

-- Create continuous aggregates for real-time analytics
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_trading_metrics
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', time) AS hour,
    symbol,
    COUNT(*) AS signal_count,
    SUM(CASE WHEN signal_type = 'buy' THEN 1 ELSE 0 END) AS buy_signals,
    SUM(CASE WHEN signal_type = 'sell' THEN 1 ELSE 0 END) AS sell_signals,
    AVG(confidence) AS avg_confidence,
    AVG(expected_return) AS avg_expected_return
FROM signals
GROUP BY hour, symbol;

-- Add continuous aggregate policy
SELECT add_continuous_aggregate_policy('hourly_trading_metrics',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create retention policies (keep data for 2 years)
SELECT add_retention_policy('market_data', INTERVAL '2 years', if_not_exists => TRUE);
SELECT add_retention_policy('news_sentiment', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('social_sentiment', INTERVAL '6 months', if_not_exists => TRUE);
SELECT add_retention_policy('features', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('predictions', INTERVAL '1 year', if_not_exists => TRUE);
SELECT add_retention_policy('performance_metrics', INTERVAL '1 year', if_not_exists => TRUE);

-- Keep orders and executions forever for audit compliance
-- No retention policy for orders, executions, positions

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO quantcli;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO quantcli;

-- Summary view
CREATE OR REPLACE VIEW trading_system_stats AS
SELECT
    (SELECT COUNT(*) FROM market_data) AS market_data_rows,
    (SELECT COUNT(DISTINCT symbol) FROM market_data) AS symbols_tracked,
    (SELECT COUNT(*) FROM signals) AS total_signals,
    (SELECT COUNT(*) FROM orders) AS total_orders,
    (SELECT COUNT(*) FROM executions) AS total_executions,
    (SELECT SUM(quantity * fill_price) FROM executions WHERE side = 'buy') AS total_capital_deployed,
    (SELECT LAST(total_value, time) FROM portfolio_performance) AS current_portfolio_value,
    (SELECT LAST(cumulative_return, time) FROM portfolio_performance) AS total_return;

-- Log initialization
INSERT INTO system_events (time, event_type, severity, message)
VALUES (NOW(), 'database_init', 'info', 'TimescaleDB initialized successfully for QuantCLI');
