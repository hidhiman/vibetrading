"""India futures (NSE / BSE F&O) backtest engine.

Trades the continuous series served for ``NIFTY-I.NFO`` / ``SENSEX-I.BFO``
(back-adjusted at every roll, so holding through expiry is modelled as a
cost-free roll). Prices are in INR per unit; position size is in units and is
rounded down to whole lots.

Market rules:
  - Lot size: per underlying, from the downloader's instrument registry
    (``VIBE_TRADING_INDIA_INSTRUMENTS``, the ``instruments.yml`` written by
    datadownloader-v1); config ``lot_size`` (number or ``{symbol: number}``)
    overrides it. Current lots apply to the whole history — NSE revises them
    periodically, so older trades are sized on today's lot.
  - Margin: config ``margin_rate`` or a default by contract type — index
    futures 12%, stock futures 20% (SPAN + exposure, approximate). As in the
    other futures engines, a target weight is the share of equity committed as
    margin, so weight 1.0 on NIFTY is roughly 8x notional exposure.
  - T+0, long and short. Price bands are not modelled.

Cost stack (discount-broker defaults; all config-driven). NOTE: SEBI/exchange
tariffs change periodically — verify ``in_*`` rates against a current broker
schedule before relying on absolute cost figures:
  - Brokerage: ₹20 per executed order                    [in_brokerage_per_order]
  - STT: 0.05% on the sell side (from 2026-04-01)         [in_stt]
  - Exchange transaction charge: NSE 0.00173%, BSE nil    [in_exchange_txn]
  - SEBI turnover fee: ₹10/crore = 0.0001%                [in_sebi_fee]
  - Stamp duty: 0.002% on the buy side                    [in_stamp_duty]
  - GST: 18% on (brokerage + exchange txn + SEBI fee)     [in_gst]
"""

from __future__ import annotations

import functools
import logging
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd

from backtest.engines.futures_base import FuturesBaseEngine

logger = logging.getLogger(__name__)

_INSTRUMENTS_ENV = "VIBE_TRADING_INDIA_INSTRUMENTS"
_SYMBOL_RE = re.compile(r"^(?P<symbol>[A-Z0-9&\-]+?)-II?\.(?P<venue>NFO|BFO)$")
_VENUE_SEGMENTS = {"NFO": "NSE_FNO", "BFO": "BSE_FNO"}
_EXCHANGE_TXN = {"NFO": 0.0000173, "BFO": 0.0}
_INDEX_MARGIN = 0.12
_STOCK_MARGIN = 0.20


@functools.lru_cache(maxsize=2)
def _registry_contracts(path: str) -> dict:
    """``(underlying, segment)`` -> ``(lot, is_index)`` from an instruments.yml."""
    import yaml

    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(path, encoding="utf-8") as handle:
        data = yaml.load(handle, Loader=loader) or {}
    contracts: dict = {}
    for symbol, entry in (data.get("INSTRUMENT") or {}).items():
        if not isinstance(entry, dict):
            continue
        for exchange in entry.values():
            futures = exchange.get("FUTURES") if isinstance(exchange, dict) else None
            if not isinstance(futures, dict):
                continue
            for contract in futures.values():
                try:
                    lot = int(float(contract.get("lot")))
                except (TypeError, ValueError):
                    continue
                key = (str(symbol).upper(), contract.get("segment"))
                contracts[key] = (lot, contract.get("instrument") == "FUTIDX")
    return contracts


def _contract(symbol: str) -> Optional[tuple[int, bool]]:
    match = _SYMBOL_RE.match(symbol.strip().upper())
    path = os.getenv(_INSTRUMENTS_ENV, "").strip()  # noqa: env-gate — opt-in registry path
    if match is None or not path or not Path(path).is_file():
        return None
    try:
        contracts = _registry_contracts(path)
    except Exception as exc:  # noqa: BLE001 — a bad registry must not stop the run
        logger.warning("india_futures: cannot read %s: %s", path, exc)
        return None
    return contracts.get((match["symbol"], _VENUE_SEGMENTS[match["venue"]]))


class IndiaFuturesEngine(FuturesBaseEngine):
    """NSE / BSE futures engine for continuous ``-I`` / ``-II`` series.

    Config keys (all optional; defaults in the module docstring):
      - lot_size: int or {symbol: int}
      - margin_rate: float fraction of notional, overrides the type default
      - slippage: default 0.0002
      - in_brokerage_per_order / in_stt / in_exchange_txn / in_sebi_fee /
        in_stamp_duty / in_gst
    """

    def __init__(self, config: dict):
        super().__init__(config)
        self._lot_override = config.get("lot_size")
        self._margin_override = config.get("margin_rate")
        self.slippage_rate: float = config.get("slippage", 0.0002)
        self.in_brokerage_per_order: float = config.get("in_brokerage_per_order", 20.0)
        self.in_stt: float = config.get("in_stt", 0.0005)
        self._exchange_txn_override = config.get("in_exchange_txn")
        self.in_sebi_fee: float = config.get("in_sebi_fee", 0.000001)
        self.in_stamp_duty: float = config.get("in_stamp_duty", 0.00002)
        self.in_gst: float = config.get("in_gst", 0.18)
        self._warned_no_lot: set[str] = set()

    def _lot(self, symbol: str) -> int:
        override = self._lot_override
        if isinstance(override, dict):
            override = override.get(symbol)
        if override:
            return max(int(override), 1)
        contract = _contract(symbol)
        if contract is not None:
            return contract[0]
        if symbol not in self._warned_no_lot:
            self._warned_no_lot.add(symbol)
            logger.warning(
                "india_futures: no lot size for %s; sizing in single units. Set %s or config lot_size.",
                symbol, _INSTRUMENTS_ENV,
            )
        return 1

    def _leverage_for_symbol(self, symbol: str) -> float:
        if self._margin_override:
            return 1.0 / float(self._margin_override)
        contract = _contract(symbol)
        is_index = contract[1] if contract is not None else False
        return 1.0 / (_INDEX_MARGIN if is_index else _STOCK_MARGIN)

    def can_execute(self, symbol: str, direction: int, bar: pd.Series) -> bool:
        """Futures: T+0, both directions; price bands are not modelled."""
        return True

    def round_size(self, raw_size: float, price: float) -> float:
        """Whole lots of the active symbol, in units."""
        lot = self._lot(self._active_symbol)
        return float(max(int(raw_size // lot), 0) * lot)

    def calc_commission(self, size: float, price: float, direction: int, is_open: bool) -> float:
        """India F&O futures cost stack (see module docstring).

        ``direction`` is the position's direction, so opening a short and
        closing a long are the sell legs.
        """
        symbol = self._active_symbol or ""
        notional = size * price
        venue = symbol.rsplit(".", 1)[-1].upper()
        txn_rate = self._exchange_txn_override
        if txn_rate is None:
            txn_rate = _EXCHANGE_TXN.get(venue, _EXCHANGE_TXN["NFO"])
        brokerage = self.in_brokerage_per_order if size > 0 else 0.0
        exchange_txn = notional * txn_rate
        sebi_fee = notional * self.in_sebi_fee
        gst = (brokerage + exchange_txn + sebi_fee) * self.in_gst
        is_sell = (direction < 0) == is_open
        tax = notional * (self.in_stt if is_sell else self.in_stamp_duty)
        return brokerage + exchange_txn + sebi_fee + gst + tax

    def apply_slippage(self, price: float, direction: int) -> float:
        """Proportional slippage (configurable)."""
        return price * (1 + direction * self.slippage_rate)

    def get_contract_multiplier(self, symbol: str) -> float:
        """Sizes are in units, so one point is one rupee per unit."""
        return 1.0
