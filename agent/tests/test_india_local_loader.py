"""Tests for the India local Parquet archive loader."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd
import pytest

import backtest.loaders.india_local_loader as india_local
from tests.loader_contract import assert_loader_contract


def _write(root: Path, exchange_dir: str, symbol: str, timeframe: str, ist_times: list[str]) -> None:
    """Write bars stamped in UTC, as the downloader archive stores them."""
    utc = pd.DatetimeIndex(ist_times).tz_localize("Asia/Kolkata").tz_convert("UTC")
    n = len(utc)
    frame = pd.DataFrame(
        {
            "datetime": utc,
            "open": [100.0 + i for i in range(n)],
            "high": [102.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [101.0 + i for i in range(n)],
            "volume": [1000.0] * n,
            "oi": [0.0] * n,
        }
    )
    year = str(utc[0].tz_convert("Asia/Kolkata").year)
    _copy(frame, root / "EQUITY" / exchange_dir / year / f"{symbol}_{timeframe}.parquet")


def _copy(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # DuckDB writes the file: the loader reads through it and pyarrow is optional.
    with duckdb.connect() as conn:
        conn.register("frame", frame)
        conn.execute(f"COPY frame TO '{path.as_posix()}' (FORMAT parquet)")


def _write_series(path: Path, ist_times, close, oi=None, volume=1000.0) -> None:
    """Write bars with the given closes (open/high/low around them), UTC-stamped."""
    utc = pd.DatetimeIndex(ist_times).tz_localize("Asia/Kolkata").tz_convert("UTC")
    close = pd.Series(close, dtype="float64").to_numpy()
    _copy(
        pd.DataFrame(
            {
                "datetime": utc,
                "open": close,
                "high": close + 1.0,
                "low": close - 1.0,
                "close": close,
                "volume": volume,
                "oi": 0.0 if oi is None else pd.Series(oi, dtype="float64").to_numpy(),
            }
        ),
        path,
    )


@pytest.fixture
def archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("VIBE_TRADING_INDIA_DATA_ROOT", str(tmp_path))
    monkeypatch.delenv("VIBE_TRADING_INDIA_DATA_MAX_STALE_DAYS", raising=False)
    monkeypatch.delenv("VIBE_TRADING_INDIA_FUTURES_START", raising=False)
    return tmp_path


def test_unavailable_without_root(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBE_TRADING_INDIA_DATA_ROOT", raising=False)
    loader = india_local.DataLoader()
    assert not loader.is_available()
    assert loader.fetch(["RELIANCE.NS"], "2025-06-02", "2025-06-03") == {}


def test_daily_bars_land_on_ist_trading_date(archive: Path) -> None:
    # Daily bars sit at IST midnight, i.e. 18:30 UTC of the previous day.
    _write(archive, "NSE_EQ", "RELIANCE", "daily", ["2025-06-02", "2025-06-03"])
    loader = india_local.DataLoader()
    assert loader.is_available()

    frames = loader.fetch(["RELIANCE.NS"], "2025-06-02", "2025-06-03")

    frame = frames["RELIANCE.NS"]
    assert_loader_contract(frame, context="india_local daily")
    assert list(frame.index) == [pd.Timestamp("2025-06-02"), pd.Timestamp("2025-06-03")]
    assert frame.attrs["adjustment"] == "split"


def test_bse_suffix_reads_bse_directory(archive: Path) -> None:
    _write(archive, "BSE_EQ", "RELIANCE", "daily", ["2025-06-02"])
    frames = india_local.DataLoader().fetch(["RELIANCE.BO", "RELIANCE.NS"], "2025-06-02", "2025-06-02")
    assert set(frames) == {"RELIANCE.BO"}


@pytest.mark.parametrize(
    ("interval", "expected_times"),
    [
        ("15m", ["09:15", "09:30", "09:45", "10:00", "10:15", "10:30"]),
        ("30m", ["09:15", "09:45", "10:15"]),
        ("1H", ["09:15", "10:15"]),
    ],
)
def test_intraday_bars_anchor_at_session_open(archive: Path, interval: str, expected_times: list[str]) -> None:
    times = [f"2025-06-02 {t}" for t in ["09:15", "09:30", "09:45", "10:00", "10:15", "10:30"]]
    _write(archive, "NSE_EQ", "RELIANCE", "15min", times)

    frame = india_local.DataLoader().fetch(["RELIANCE.NS"], "2025-06-02", "2025-06-02", interval=interval)["RELIANCE.NS"]

    assert_loader_contract(frame, context=f"india_local {interval}")
    assert [ts.strftime("%H:%M") for ts in frame.index] == expected_times


def test_resampled_bar_aggregates_ohlcv(archive: Path) -> None:
    _write(archive, "NSE_EQ", "RELIANCE", "15min", ["2025-06-02 09:15", "2025-06-02 09:30"])
    bar = india_local.DataLoader().fetch(["RELIANCE.NS"], "2025-06-02", "2025-06-02", interval="30m")["RELIANCE.NS"].iloc[0]
    assert (bar.open, bar.high, bar.low, bar.close, bar.volume) == (100.0, 103.0, 99.0, 102.0, 2000.0)


def test_stale_archive_defers_to_next_source(archive: Path) -> None:
    _write(archive, "NSE_EQ", "RELIANCE", "daily", ["2025-06-02", "2025-06-03"])
    loader = india_local.DataLoader()
    assert loader.fetch(["RELIANCE.NS"], "2025-06-01", "2025-06-30") == {}
    assert "RELIANCE.NS" in loader.fetch(["RELIANCE.NS"], "2025-06-01", "2025-06-06")


def test_missing_symbol_and_foreign_suffix_are_skipped(archive: Path) -> None:
    _write(archive, "NSE_EQ", "RELIANCE", "daily", ["2025-06-02"])
    frames = india_local.DataLoader().fetch(["NOTREAL.NS", "AAPL.US"], "2025-06-02", "2025-06-02")
    assert frames == {}


# ── Indices ──


def test_index_reads_yahoo_alias_and_archive_name(archive: Path) -> None:
    days = pd.bdate_range("2025-06-02", periods=3)
    _write_series(archive / "INDEX" / "2025" / "NIFTY_daily.parquet", days, [24000.0, 24100.0, 24200.0])
    _write_series(archive / "INDEX" / "2025" / "NIFTY_AUTO_daily.parquet", days, [23000.0, 23100.0, 23200.0])

    frames = india_local.DataLoader().fetch(
        ["^NSEI", "^NIFTY-AUTO", "^NOTANINDEX"], str(days[0].date()), str(days[-1].date())
    )

    assert set(frames) == {"^NSEI", "^NIFTY-AUTO"}
    assert_loader_contract(frames["^NSEI"], context="india_local index")
    assert list(frames["^NSEI"]["close"]) == [24000.0, 24100.0, 24200.0]
    assert list(frames["^NIFTY-AUTO"]["close"]) == [23000.0, 23100.0, 23200.0]
    assert "adjustment" not in frames["^NSEI"].attrs


# ── Futures ──

# 60 sessions; the near contract expires after session 29, when the next-month
# contract becomes the near month and a fresh far month becomes the next.
_DAYS = pd.bdate_range("2025-01-01", periods=60)
_ROLL = 29
_BASE = pd.Series([100.0 + 0.1 * i for i in range(len(_DAYS))])
_OLD = _BASE.index <= _ROLL


def _futures(root: Path, venue_dir: str, symbol: str, label: str, close, oi, timeframe: str = "daily", days=_DAYS) -> None:
    path = root / "FUTURES" / venue_dir / "2025" / f"{symbol}_{label}_{timeframe}.parquet"
    _write_series(path, days, close, oi=oi)


def _nifty_archive(root: Path) -> None:
    """Three ranks, each a carry of 1% over the one before."""
    near = _BASE.where(_OLD, _BASE * 1.01)
    nxt = (_BASE * 1.01).where(_OLD, _BASE * 1.02)
    far = (_BASE * 1.02).where(_OLD, _BASE * 1.03)
    _futures(root, "NSE_FNO", "NIFTY", "2025-01-30", near, [10_000.0] * 60)
    # A second near-month label covering the later sessions: same series.
    _futures(root, "NSE_FNO", "NIFTY", "2025-02-27", near[30:], [10_000.0] * 30, days=_DAYS[30:])
    # The next month's open interest collapses after the roll: a fresh contract.
    _futures(root, "NSE_FNO", "NIFTY", "2025-03-27", nxt, [5_000.0] * 30 + [500.0] * 30)
    _futures(root, "NSE_FNO", "NIFTY", "2025-04-24", far, [100.0] * 60)


def _fetch(code: str, interval: str = "1D", start=_DAYS[0], end=_DAYS[-1]) -> pd.DataFrame:
    frames = india_local.DataLoader().fetch([code], str(start.date()), str(end.date()), interval=interval)
    return frames.get(code)


def test_near_month_is_ratio_back_adjusted_at_the_roll(archive: Path) -> None:
    _nifty_archive(archive)

    frame = _fetch("NIFTY-I.NFO")

    assert_loader_contract(frame, context="india_local futures")
    assert len(frame) == len(_DAYS)
    # Sessions before the roll are scaled onto the contract held after it,
    # so the series is one smooth carry with no jump at expiry.
    assert frame["close"].to_numpy() == pytest.approx((_BASE * 1.01).to_numpy())
    # The latest bar keeps its traded price.
    assert frame["close"].iloc[-1] == pytest.approx(_BASE.iloc[-1] * 1.01)


def test_next_month_is_adjusted_onto_the_far_month(archive: Path) -> None:
    _nifty_archive(archive)
    frame = _fetch("NIFTY-II.NFO")
    assert frame["close"].to_numpy() == pytest.approx((_BASE * 1.02).to_numpy())


def test_intraday_futures_take_their_sessions_factor(archive: Path) -> None:
    _nifty_archive(archive)
    session = _DAYS[10]
    times = [session + pd.Timedelta(hours=9, minutes=15), session + pd.Timedelta(hours=9, minutes=30)]
    _futures(archive, "NSE_FNO", "NIFTY", "2025-01-30", [50.0, 51.0], [10_000.0] * 2, timeframe="15min", days=times)

    frame = _fetch("NIFTY-I.NFO", interval="15m", start=session, end=session)

    assert frame["close"].to_numpy() == pytest.approx([50.5, 51.51])


def test_futures_start_floor(archive: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _nifty_archive(archive)
    monkeypatch.setenv("VIBE_TRADING_INDIA_FUTURES_START", "2025-02-03")
    frame = _fetch("NIFTY-I.NFO")
    assert frame.index.min() == pd.Timestamp("2025-02-03")


def test_stock_future_is_adjusted_for_a_bonus(archive: Path) -> None:
    # 1:1 bonus at session 20: the future halves, the adjusted equity does not.
    ex_date = 20
    before = _BASE.index < ex_date
    near = (_BASE * 2).where(before, _BASE)
    _futures(archive, "NSE_FNO", "RELIANCE", "2025-01-30", near, [10_000.0] * 60)
    _futures(archive, "NSE_FNO", "RELIANCE", "2025-02-27", near * 1.01, [5_000.0] * 60)
    _write_series(archive / "EQUITY" / "NSE_EQ" / "2025" / "RELIANCE_daily.parquet", _DAYS, _BASE)

    frame = _fetch("RELIANCE-I.NFO")

    assert frame["close"].to_numpy() == pytest.approx(_BASE.to_numpy())
    assert frame["volume"].to_numpy() == pytest.approx([2000.0] * ex_date + [1000.0] * (60 - ex_date))


def test_unknown_rank_and_single_series_are_not_served(archive: Path) -> None:
    _nifty_archive(archive)
    _futures(archive, "NSE_FNO", "TCS", "2025-01-30", _BASE, [10_000.0] * 60)

    assert _fetch("NIFTY-III.NFO") is None
    assert _fetch("NIFTY-I.BFO") is None
    # One rank only: rolls cannot be found, so the series is not served raw.
    assert _fetch("TCS-I.NFO") is None
