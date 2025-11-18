"""
Unit tests for RateLimiter class.
"""

import pytest
import time
import threading
from src.data.providers.base import RateLimiter


class TestRateLimiter:
    """Test suite for RateLimiter."""

    def test_init(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(calls=10, period=60)

        assert limiter.calls == 10
        assert limiter.period == 60
        assert limiter.tokens == 10
        assert limiter.last_update > 0

    def test_acquire_success(self):
        """Test successful token acquisition."""
        limiter = RateLimiter(calls=10, period=60)

        # First 10 calls should succeed
        for i in range(10):
            assert limiter.acquire(blocking=False) is True, f"Call {i+1} should succeed"

        # 11th call should fail (non-blocking)
        assert limiter.acquire(blocking=False) is False

    def test_token_refill(self):
        """Test that tokens refill over time."""
        limiter = RateLimiter(calls=10, period=1)  # 10 calls per second

        # Consume all tokens
        for _ in range(10):
            assert limiter.acquire(blocking=False) is True

        # No tokens left
        assert limiter.acquire(blocking=False) is False

        # Wait for token refill (0.2 seconds = 2 tokens)
        time.sleep(0.2)

        # Should have ~2 tokens now
        assert limiter.acquire(blocking=False) is True
        assert limiter.acquire(blocking=False) is True

        # Should be out again
        assert limiter.acquire(blocking=False) is False

    def test_blocking_acquire(self):
        """Test blocking token acquisition."""
        limiter = RateLimiter(calls=2, period=1)

        # Consume both tokens
        assert limiter.acquire(blocking=False) is True
        assert limiter.acquire(blocking=False) is True

        # Third call blocks briefly
        start = time.time()
        assert limiter.acquire(blocking=True) is True
        elapsed = time.time() - start

        # Should have waited approximately 0.5 seconds for next token
        assert 0.3 < elapsed < 0.7, f"Expected ~0.5s wait, got {elapsed}s"

    def test_thread_safety(self):
        """Test thread safety of rate limiter."""
        limiter = RateLimiter(calls=100, period=1)
        tokens_acquired = []
        lock = threading.Lock()

        def acquire_tokens():
            """Try to acquire 20 tokens."""
            local_count = 0
            for _ in range(20):
                if limiter.acquire(blocking=False):
                    local_count += 1
            with lock:
                tokens_acquired.append(local_count)

        # Create 10 threads, each trying to acquire 20 tokens (200 total)
        threads = [threading.Thread(target=acquire_tokens) for _ in range(10)]

        # Start all threads
        for t in threads:
            t.start()

        # Wait for all threads to complete
        for t in threads:
            t.join()

        # Total acquired should not exceed 100 (the limit)
        total_acquired = sum(tokens_acquired)
        assert total_acquired <= 100, f"Too many tokens acquired: {total_acquired}/100"
        assert total_acquired >= 95, f"Too few tokens acquired: {total_acquired}/100"

    def test_rate_limit_calculation(self):
        """Test accurate rate limiting over time."""
        limiter = RateLimiter(calls=5, period=1)

        # Acquire all 5 tokens
        for _ in range(5):
            assert limiter.acquire(blocking=False) is True

        # Should be rate limited now
        assert limiter.acquire(blocking=False) is False

        # Wait exactly 1 second
        time.sleep(1.0)

        # Should have all 5 tokens back
        for i in range(5):
            result = limiter.acquire(blocking=False)
            assert result is True, f"Token {i+1} should be available"

    def test_high_frequency_limiting(self):
        """Test rate limiting with high frequency calls."""
        limiter = RateLimiter(calls=10, period=0.1)  # 10 calls per 100ms

        successes = 0
        for _ in range(20):
            if limiter.acquire(blocking=False):
                successes += 1

        # Should get exactly 10 successes
        assert successes == 10

    def test_token_ceiling(self):
        """Test that tokens don't exceed maximum."""
        limiter = RateLimiter(calls=5, period=1)

        # Wait for tokens to potentially accumulate
        time.sleep(2)

        # Should still only have 5 tokens max
        successes = 0
        for _ in range(10):
            if limiter.acquire(blocking=False):
                successes += 1

        assert successes == 5, f"Expected 5 tokens, got {successes}"

    def test_fractional_tokens(self):
        """Test handling of fractional token refill."""
        limiter = RateLimiter(calls=10, period=1)

        # Consume all tokens
        for _ in range(10):
            limiter.acquire(blocking=False)

        # Wait for fractional token refill (0.05s = 0.5 tokens)
        time.sleep(0.05)

        # Should not have a full token yet
        assert limiter.acquire(blocking=False) is False

        # Wait a bit more (total 0.15s = 1.5 tokens)
        time.sleep(0.10)

        # Should have at least 1 token now
        assert limiter.acquire(blocking=False) is True

    def test_concurrent_blocking_acquire(self):
        """Test multiple threads blocking on rate limit."""
        limiter = RateLimiter(calls=2, period=1)
        results = []
        lock = threading.Lock()

        def blocking_acquire():
            """Acquire with blocking."""
            result = limiter.acquire(blocking=True)
            with lock:
                results.append((result, time.time()))

        # Start 5 threads simultaneously
        threads = [threading.Thread(target=blocking_acquire) for _ in range(5)]
        start = time.time()

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        elapsed = time.time() - start

        # All should succeed
        assert all(r[0] for r in results)

        # Should take at least 1.5 seconds (need to wait for tokens)
        assert elapsed >= 1.0, f"Expected >=1.0s, got {elapsed}s"

    def test_zero_calls(self):
        """Test behavior with zero calls allowed."""
        limiter = RateLimiter(calls=0, period=1)

        # Should never acquire
        assert limiter.acquire(blocking=False) is False

    @pytest.mark.slow
    def test_long_period_limiting(self):
        """Test rate limiting over longer period."""
        limiter = RateLimiter(calls=10, period=2)

        # Acquire all 10
        for _ in range(10):
            assert limiter.acquire(blocking=False) is True

        # Should be limited
        assert limiter.acquire(blocking=False) is False

        # Wait half period
        time.sleep(1.0)

        # Should have ~5 tokens
        successes = 0
        for _ in range(10):
            if limiter.acquire(blocking=False):
                successes += 1

        assert 4 <= successes <= 6, f"Expected ~5 tokens, got {successes}"
