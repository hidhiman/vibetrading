"""India local loader: NSE/BSE equities, indices and futures from a local Parquet archive.

Reads a downloader archive (``VIBE_TRADING_INDIA_DATA_ROOT``) laid out as::

    <root>/EQUITY/{NSE_EQ,BSE_EQ}/<year>/<SYMBOL>_<timeframe>.parquet
    <root>/INDEX/<year>/<NAME>_<timeframe>.parquet
    <root>/FUTURES/{NSE_FNO,BSE_FNO}/<year>/<SYMBOL>_<label>_<timeframe>.parquet

where ``timeframe`` is ``1min``, ``5min``, ``15min`` or ``daily`` (hourly files
exist but are clock-aligned, so hours are built from 15min) and each file holds
``datetime`` (tz-aware) plus ``open``, ``high``, ``low``, ``close``, ``volume``.
The loader is unavailable without the root.

Symbols:

* Equities follow the Yahoo chain: ``RELIANCE.NS`` reads ``NSE_EQ``,
  ``RELIANCE.BO`` reads ``BSE_EQ``. Prices are split/bonus adjusted but not
  dividend adjusted (caliber ``split``).
* Indices use Yahoo tickers (``^NSEI``, ``^NSEBANK``, ``^BSESN``, ...) or
  ``^<archive name>`` with ``-`` for ``_`` (``^NIFTY-AUTO``).
* Futures are continuous by rank: ``NIFTY-I.NFO`` (near month), ``NIFTY-II.NFO``
  (next month); ``.NFO`` reads ``NSE_FNO``, ``.BFO`` reads ``BSE_FNO``.

Futures files are not single contracts. Each one is the broker's continuous
series for the rank its contract held when it was downloaded, so the ``label``
in the name identifies a rank, not the bars inside. The loader groups labels
whose closes agree, orders the groups by open interest (near > next), finds each
roll from the next-month series' open interest collapsing overnight, and
ratio back-adjusts the series at every roll so a held position shows no jump
there. Stock futures are also ratio-adjusted for bonuses and splits, found where
the futures move departs from the adjusted equity series by more than
``_CORPORATE_ACTION_MOVE``. The latest bars keep their traded prices.

Futures bars start at ``VIBE_TRADING_INDIA_FUTURES_START`` (default 2017-01-01).
Earlier broker series are corrupt around expiries (a near-month series showing
a dead contract the day after expiry, all three ranks identical for days), so
rolls cannot be found and the day after each NIFTY expiry carried a +0.25% to
+0.75% residual jump against spot through 2016; from 2017 it is within noise.

Bars are returned on IST wall-clock time (daily bars on the IST trading date),
like the yfinance loader. An archive that stops short of the requested window
returns nothing, so the chain can fall through to the next source instead of
serving a series that silently ends days early.
``VIBE_TRADING_INDIA_DATA_MAX_STALE_DAYS`` (default 5 calendar days) sets how
far the last bar may trail ``end_date``.
"""

from __future__ import annotations

import functools
import logging
import os
import re
from fractions import Fraction
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from backtest.loaders.base import validate_date_range, validate_ohlc
from backtest.loaders.registry import register

logger = logging.getLogger(__name__)

_ROOT_ENV = "VIBE_TRADING_INDIA_DATA_ROOT"
_STALE_ENV = "VIBE_TRADING_INDIA_DATA_MAX_STALE_DAYS"
_DEFAULT_MAX_STALE_DAYS = 5
_FUTURES_START_ENV = "VIBE_TRADING_INDIA_FUTURES_START"
_DEFAULT_FUTURES_START = "2017-01-01"

_EQUITY_DIRS = {".NS": "NSE_EQ", ".BO": "BSE_EQ"}
_FNO_DIRS = {"NFO": "NSE_FNO", "BFO": "BSE_FNO"}
_FNO_EQUITY_DIRS = {"NSE_FNO": "NSE_EQ", "BSE_FNO": "BSE_EQ"}
_FUTURES_RE = re.compile(r"^(?P<symbol>[A-Z0-9&\-]+?)-(?P<rank>II?)\.(?P<venue>NFO|BFO)$")

# Yahoo index tickers -> archive index names.
_INDEX_ALIASES = {
    "NSEI": "NIFTY",
    "NSEBANK": "BANKNIFTY",
    "BSESN": "SENSEX",
    "INDIAVIX": "INDIA_VIX",
    "CNXIT": "NIFTYIT",
    "NSMIDCP": "NIFTY_NEXT_50",
    "NSEMDCP50": "NIFTYMCAP50",
    "CNXAUTO": "NIFTY_AUTO",
    "CNXPHARMA": "NIFTY_PHARMA",
    "CNXFMCG": "NIFTY_FMCG",
    "CNXMETAL": "NIFTY_METAL",
    "CNXREALTY": "NIFTY_REALTY",
    "CNXENERGY": "NIFTY_ENERGY",
    "CNXPSUBANK": "NIFTY_PSU_BANK",
    "CNXMEDIA": "NIFTY_MEDIA",
    "CNXINFRA": "NIFTYINFRA",
    "CNXPSE": "NIFTYPSE",
    "CNX100": "NIFTY_100",
    "CNX200": "NIFTY_200",
    "CRSLDX": "NIFTY_500",
    "CNXSC": "NIFTY_SMALLCAP_100",
}

# Project interval -> (archive timeframe, pandas resample rule or None).
# Resampled bars are anchored at the 09:15 IST session open. Hours come from
# 15min bars because the archive's hourly files are clock-aligned (09:00, 10:00).
_INTERVALS = {
    "1m": ("1min", None),
    "5m": ("5min", None),
    "15m": ("15min", None),
    "30m": ("15min", "30min"),
    "1H": ("15min", "1h"),
    "1h": ("15min", "1h"),
    "4H": ("15min", "4h"),
    "4h": ("15min", "4h"),
    "1D": ("daily", None),
    "1d": ("daily", None),
}

_SESSION_OPEN = pd.Timedelta(hours=9, minutes=15)

_OHLCV = ["open", "high", "low", "close", "volume"]
_PRICES = ["open", "high", "low", "close"]
_OHLCV_AGG = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}

# Labels whose closes agree on this share of common days are one series.
_SAME_SERIES_SHARE = 0.95
# The next-month series' open interest falls below this share of the previous
# day's when its contract becomes the near month (the new next month is fresh).
_ROLL_OI_DROP = 0.5
# A futures move this far from the equity's adjusted move is a corporate action.
_CORPORATE_ACTION_MOVE = 0.15


def _root() -> Optional[Path]:
    raw = os.getenv(_ROOT_ENV, "").strip()  # noqa: env-gate — opt-in archive path
    if not raw:
        return None
    path = Path(raw).expanduser()
    return path if path.is_dir() else None


def _max_stale_days() -> int:
    raw = os.getenv(_STALE_ENV, "").strip()  # noqa: env-gate — opt-in tuning
    try:
        return max(0, int(raw)) if raw else _DEFAULT_MAX_STALE_DAYS
    except ValueError:
        return _DEFAULT_MAX_STALE_DAYS


def _futures_start() -> pd.Timestamp:
    raw = os.getenv(_FUTURES_START_ENV, "").strip()  # noqa: env-gate — opt-in tuning
    try:
        return pd.Timestamp(raw or _DEFAULT_FUTURES_START)
    except ValueError:
        return pd.Timestamp(_DEFAULT_FUTURES_START)


def _query(sql: str, params: list) -> pd.DataFrame:
    import duckdb

    with duckdb.connect() as conn:
        return conn.execute(sql, params).df()


def _read_bars(files: List[Path]) -> pd.DataFrame:
    """OHLCV bars from ``files`` on IST wall-clock time, one row per timestamp.

    Where files overlap (labels of one futures rank) the busiest row wins
    whole, so a bar never mixes columns from rows that disagree.
    """
    frame = _query(
        "SELECT timezone('Asia/Kolkata', datetime) AS trade_date, open, high, low, close, volume "
        "FROM read_parquet(?, union_by_name = true, filename = true) "
        "QUALIFY row_number() OVER (PARTITION BY datetime ORDER BY volume DESC NULLS LAST, filename) = 1",
        [[str(path) for path in files]],
    )
    return frame.set_index("trade_date").sort_index()


def _year_files(base: Path, start: pd.Timestamp, end: pd.Timestamp, name: str) -> List[Path]:
    return [
        path
        for year in range(start.year, end.year + 1)
        if (path := base / str(year) / name).is_file()
    ]


def _snap_ratio(ratio: float) -> float:
    """Snap a measured split/bonus factor to the nearest simple fraction."""
    fraction = Fraction(ratio).limit_denominator(10)
    return float(fraction) if abs(float(fraction) / ratio - 1) < 0.03 else ratio


# ── Indices ──


@functools.lru_cache(maxsize=4)
def _index_names(index_dir: str) -> Dict[str, str]:
    """Upper-cased lookup key -> archive index name, over every year directory."""
    names: Dict[str, str] = {}
    for year_dir in sorted(Path(index_dir).glob("[0-9][0-9][0-9][0-9]")):
        for path in year_dir.glob("*_daily.parquet"):
            name = path.name[: -len("_daily.parquet")]
            names.setdefault(name.upper(), name)
    return names


def _index_name(root: Path, code: str) -> Optional[str]:
    key = code.strip().upper()[1:]
    names = _index_names(str(root / "INDEX"))
    for candidate in (_INDEX_ALIASES.get(key), key, key.replace("-", "_"), key.replace("-", " ")):
        if candidate and candidate.upper() in names:
            return names[candidate.upper()]
    return None


# ── Futures ──


def _futures_daily_files(root: Path, venue_dir: str, symbol: str) -> tuple:
    files = sorted((root / "FUTURES" / venue_dir).glob(f"[0-9][0-9][0-9][0-9]/{symbol}_????-??-??_daily.parquet"))
    return tuple((str(path), path.stat().st_mtime_ns) for path in files)


@functools.lru_cache(maxsize=64)
def _futures_model(root: str, venue_dir: str, symbol: str, revision: tuple) -> Optional[dict]:
    """Rank groups and per-date back-adjustment factors for one underlying.

    Returns ``{"labels": [near_labels, next_labels, ...], "factors": [df, df]}``
    where each factor frame is indexed by IST date with ``price`` (multiply the
    raw prices) and ``volume`` (divide the raw volume) columns for that rank.
    ``revision`` keys the cache on the daily files' modification times.
    """
    if not revision:
        return None
    daily = _query(
        "SELECT regexp_extract(filename, '_(\\d{4}-\\d{2}-\\d{2})_daily', 1) AS label, "
        "timezone('Asia/Kolkata', datetime)::DATE AS day, close, oi "
        "FROM read_parquet(?, filename = true)",
        [[path for path, _ in revision]],
    )
    daily["day"] = pd.to_datetime(daily["day"])
    closes = daily.pivot_table(index="day", columns="label", values="close", aggfunc="last")
    interest = daily.pivot_table(index="day", columns="label", values="oi", aggfunc="last")

    groups: List[List[str]] = []
    for label in closes.columns:
        for group in groups:
            both = closes[[label, group[0]]].dropna()
            if len(both) >= 20 and (both[label] == both[group[0]]).mean() >= _SAME_SERIES_SHARE:
                group.append(label)
                break
        else:
            groups.append([label])
    if len(groups) < 2:
        logger.info("india_local: %s %s has no next-month series; rolls cannot be adjusted", venue_dir, symbol)
        return None

    def first_valid(frame: pd.DataFrame, labels: List[str]) -> pd.Series:
        return frame[labels].bfill(axis=1).iloc[:, 0]

    groups.sort(key=lambda labels: -first_valid(interest, labels).median())
    close = [first_valid(closes, labels) for labels in groups]
    oi = [first_valid(interest, labels) for labels in groups]

    next_oi = oi[1].dropna()
    drop = next_oi.shift(-1) / next_oi.where(next_oi > 0)
    rolls = set(drop.index[drop < _ROLL_OI_DROP])

    actions = _corporate_actions(Path(root), venue_dir, symbol, close[0].dropna())

    factors = []
    for rank in range(min(2, len(groups))):
        days = close[rank].dropna().index
        price, volume = 1.0, 1.0
        rows = []
        for i in range(len(days) - 1, -1, -1):
            day = days[i]
            rows.append((day, price, volume))
            if i == 0:
                break
            ratio = actions.get(day)
            if ratio is not None:
                price *= ratio
                volume *= ratio
            previous = days[i - 1]
            if previous in rolls:
                successor = close[rank + 1].get(previous) if rank + 1 < len(close) else None
                current = close[rank].get(previous)
                if successor and current and pd.notna(successor) and pd.notna(current):
                    price *= successor / current
        factor = pd.DataFrame(rows, columns=["day", "price", "volume"]).set_index("day").sort_index()
        factors.append(factor)

    logger.debug(
        "india_local: %s %s groups=%s rolls=%d corporate_actions=%s",
        venue_dir, symbol, groups, len(rolls), actions,
    )
    return {"labels": groups[:2], "factors": factors}


def _corporate_actions(root: Path, venue_dir: str, symbol: str, near: pd.Series) -> Dict[pd.Timestamp, float]:
    """Ex-dates -> price ratio (new/old), from the adjusted equity series."""
    equity_dir = root / "EQUITY" / _FNO_EQUITY_DIRS[venue_dir]
    files = sorted(equity_dir.glob(f"[0-9][0-9][0-9][0-9]/{symbol}_daily.parquet"))
    if not files or near.empty:
        return {}
    equity = _query(
        "SELECT timezone('Asia/Kolkata', datetime)::DATE AS day, any_value(close) AS close "
        "FROM read_parquet(?) GROUP BY 1",
        [[str(path) for path in files]],
    )
    equity["day"] = pd.to_datetime(equity["day"])
    joined = pd.DataFrame({"future": near, "equity": equity.set_index("day")["close"]}).dropna()
    relative = (joined["future"] / joined["future"].shift(1)) / (joined["equity"] / joined["equity"].shift(1))
    moved = relative[(relative - 1).abs() > _CORPORATE_ACTION_MOVE]
    return {day: _snap_ratio(float(value)) for day, value in moved.items()}


@register
class DataLoader:
    """Local Parquet archive for NSE/BSE equities, indices and futures."""

    name = "india_local"
    markets = {"india_equity", "index", "india_futures"}
    requires_auth = False

    def is_available(self) -> bool:
        return _root() is not None

    def fetch(
        self,
        codes: List[str],
        start_date: str,
        end_date: str,
        *,
        interval: str = "1D",
        fields: Optional[List[str]] = None,
    ) -> Dict[str, pd.DataFrame]:
        """Fetch OHLCV bars for each code from the archive.

        Args:
            codes: ``RELIANCE.NS`` / ``RELIANCE.BO``, ``^NSEI``, ``NIFTY-I.NFO``.
            start_date: YYYY-MM-DD.
            end_date: YYYY-MM-DD (inclusive of the whole day).
            interval: Bar size; see ``_INTERVALS``.
            fields: Ignored.

        Returns:
            Mapping code -> OHLCV DataFrame; codes without archive coverage are
            omitted so the fallback chain can serve them.
        """
        validate_date_range(start_date, end_date)
        root = _root()
        spec = _INTERVALS.get(interval)
        if root is None or spec is None:
            if spec is None:
                logger.debug("india_local: unsupported interval %r", interval)
            return {}

        result: Dict[str, pd.DataFrame] = {}
        for code in codes:
            try:
                frame = self._fetch_one(root, code, start_date, end_date, spec)
            except Exception as exc:  # noqa: BLE001 — per-symbol fallback
                logger.warning("india_local failed for %s: %s", code, exc)
                continue
            if frame is not None and not frame.empty:
                result[code] = frame
        return result

    def _fetch_one(
        self,
        root: Path,
        code: str,
        start_date: str,
        end_date: str,
        spec: tuple[str, Optional[str]],
    ) -> Optional[pd.DataFrame]:
        timeframe, rule = spec
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
        upper = code.strip().upper()
        adjustment: Optional[str] = None
        factors: Optional[pd.DataFrame] = None

        futures = _FUTURES_RE.match(upper)
        if futures:
            venue_dir = _FNO_DIRS[futures["venue"]]
            symbol = futures["symbol"]
            model = _futures_model(str(root), venue_dir, symbol, _futures_daily_files(root, venue_dir, symbol))
            rank = len(futures["rank"]) - 1
            if model is None or rank >= len(model["labels"]):
                return None
            start = max(start, _futures_start())
            if start >= end:
                return None
            base = root / "FUTURES" / venue_dir
            files = [
                path
                for label in model["labels"][rank]
                for path in _year_files(base, start, end, f"{symbol}_{label}_{timeframe}.parquet")
            ]
            factors = model["factors"][rank]
        elif upper.startswith("^"):
            name = _index_name(root, upper)
            if name is None:
                return None
            files = _year_files(root / "INDEX", start, end, f"{name}_{timeframe}.parquet")
        else:
            suffix = next((s for s in _EQUITY_DIRS if upper.endswith(s) and len(upper) > len(s)), None)
            if suffix is None:
                return None
            base = root / "EQUITY" / _EQUITY_DIRS[suffix]
            files = _year_files(base, start, end, f"{upper[: -len(suffix)]}_{timeframe}.parquet")
            # The archive adjusts equities for splits and bonuses but not cash
            # dividends: its closes match Yahoo's split-adjusted Close.
            adjustment = "split"
        if not files:
            return None

        frame = _read_bars(files)
        if timeframe == "daily":
            frame.index = frame.index.normalize()
        frame = frame[(frame.index >= start) & (frame.index < end)]
        if frame.empty:
            return None

        last_bar = frame.index.max().normalize()
        if last_bar < pd.Timestamp(end_date) - pd.Timedelta(days=_max_stale_days()):
            logger.info(
                "india_local: %s ends %s, before requested %s; deferring to the next source",
                code, last_bar.date(), end_date,
            )
            return None

        frame = frame[_OHLCV].apply(pd.to_numeric, errors="coerce")
        frame = frame.dropna(subset=_PRICES)
        frame["volume"] = frame["volume"].fillna(0.0)
        if factors is not None:
            frame = _apply_factors(frame, factors)
        if rule is not None:
            offset = _SESSION_OPEN % pd.Timedelta(rule)
            frame = frame.resample(rule, offset=offset).agg(_OHLCV_AGG)
            frame = frame.dropna(subset=_PRICES)
        frame = validate_ohlc(frame).astype("float64")
        frame.index.name = "trade_date"
        if adjustment is not None:
            frame.attrs["adjustment"] = adjustment
        return frame


def _apply_factors(frame: pd.DataFrame, factors: pd.DataFrame) -> pd.DataFrame:
    """Scale each bar by its trading day's back-adjustment factors.

    A day after the last daily bar (intraday files refreshed first) keeps its
    traded prices; a day the daily series lacks takes the next known day's.
    """
    days = frame.index.normalize()
    aligned = factors.reindex(factors.index.union(days.unique())).bfill().fillna(1.0).reindex(days)
    frame = frame.copy()
    frame[_PRICES] = frame[_PRICES].mul(aligned["price"].to_numpy(), axis=0)
    frame["volume"] = frame["volume"] / aligned["volume"].to_numpy()
    return frame
