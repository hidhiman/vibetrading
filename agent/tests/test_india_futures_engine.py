"""Tests for IndiaFuturesEngine: lot sizing, margin and the F&O cost stack."""

from __future__ import annotations

from pathlib import Path

import pytest

from backtest.engines.india_futures import IndiaFuturesEngine

_INSTRUMENTS = """\
INSTRUMENT:
  NIFTY:
    NSE:
      FUTURES:
        '2026-10-27': {segment: NSE_FNO, instrument: FUTIDX, lot: 65}
  RELIANCE:
    NSE:
      FUTURES:
        '2026-10-27': {segment: NSE_FNO, instrument: FUTSTK, lot: 500}
  SENSEX:
    BSE:
      FUTURES:
        '2026-10-29': {segment: BSE_FNO, instrument: FUTIDX, lot: 20}
"""


@pytest.fixture
def instruments(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "instruments.yml"
    path.write_text(_INSTRUMENTS, encoding="utf-8")
    monkeypatch.setenv("VIBE_TRADING_INDIA_INSTRUMENTS", str(path))
    return path


def _engine(symbol: str, **config) -> IndiaFuturesEngine:
    engine = IndiaFuturesEngine({"initial_cash": 1_000_000, "codes": [symbol], **config})
    engine._active_symbol = symbol
    return engine


def test_size_rounds_down_to_registry_lots(instruments: Path) -> None:
    assert _engine("NIFTY-I.NFO").round_size(200, 25_000.0) == 195.0
    assert _engine("RELIANCE-II.NFO").round_size(999, 1_400.0) == 500.0
    assert _engine("SENSEX-I.BFO").round_size(19, 80_000.0) == 0.0


def test_config_lot_size_overrides_registry(instruments: Path) -> None:
    assert _engine("NIFTY-I.NFO", lot_size={"NIFTY-I.NFO": 75}).round_size(200, 25_000.0) == 150.0
    assert _engine("NIFTY-I.NFO", lot_size=50).round_size(120, 25_000.0) == 100.0


def test_unknown_lot_sizes_in_units(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("VIBE_TRADING_INDIA_INSTRUMENTS", raising=False)
    assert _engine("NIFTY-I.NFO").round_size(200.7, 25_000.0) == 200.0


def test_leverage_by_contract_type(instruments: Path) -> None:
    assert _engine("NIFTY-I.NFO")._leverage_for_symbol("NIFTY-I.NFO") == pytest.approx(1 / 0.12)
    assert _engine("RELIANCE-I.NFO")._leverage_for_symbol("RELIANCE-I.NFO") == pytest.approx(1 / 0.20)
    assert _engine("NIFTY-I.NFO", margin_rate=0.25)._leverage_for_symbol("NIFTY-I.NFO") == pytest.approx(4.0)


def test_stt_only_on_the_sell_leg() -> None:
    engine = _engine("NIFTY-I.NFO")
    notional = 65 * 25_000.0
    # Brokerage + exchange charge + SEBI fee, plus 18% GST on all three.
    fixed = (20.0 + notional * (0.0000173 + 0.000001)) * 1.18
    buy = fixed + notional * 0.00002
    sell = fixed + notional * 0.0005

    assert engine.calc_commission(65, 25_000.0, direction=1, is_open=True) == pytest.approx(buy)
    assert engine.calc_commission(65, 25_000.0, direction=1, is_open=False) == pytest.approx(sell)
    assert engine.calc_commission(65, 25_000.0, direction=-1, is_open=True) == pytest.approx(sell)
    assert engine.calc_commission(65, 25_000.0, direction=-1, is_open=False) == pytest.approx(buy)


def test_bse_has_no_exchange_transaction_charge() -> None:
    nse = _engine("SENSEX-I.NFO").calc_commission(20, 80_000.0, direction=1, is_open=True)
    bse = _engine("SENSEX-I.BFO").calc_commission(20, 80_000.0, direction=1, is_open=True)
    assert nse - bse == pytest.approx(20 * 80_000.0 * 0.0000173 * 1.18)


def test_points_are_rupees_per_unit() -> None:
    engine = _engine("NIFTY-I.NFO")
    assert engine.get_contract_multiplier("NIFTY-I.NFO") == 1.0
    assert engine._calc_pnl("NIFTY-I.NFO", -1, 65, 25_000.0, 24_900.0) == pytest.approx(6_500.0)


def test_runner_routes_continuous_futures_to_the_engine() -> None:
    from backtest.engines.composite import CompositeEngine
    from backtest.runner import _select_market_engine

    config = {"initial_cash": 1_000_000}
    assert isinstance(_select_market_engine("auto", config, ["NIFTY-I.NFO", "SENSEX-II.BFO"]), IndiaFuturesEngine)
    mixed = _select_market_engine("auto", config, ["NIFTY-I.NFO", "RELIANCE.NS"])
    assert isinstance(mixed, CompositeEngine)
    assert isinstance(mixed._rule_engines["india_futures"], IndiaFuturesEngine)
