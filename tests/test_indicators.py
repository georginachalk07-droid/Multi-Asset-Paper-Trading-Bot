"""Indicator correctness against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from core import indicators


@pytest.fixture
def simple_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "high": [11.0, 12.0, 13.0, 12.5, 13.5],
            "low": [9.0, 10.5, 11.0, 11.5, 12.0],
            "close": [10.0, 11.5, 12.0, 12.0, 13.0],
        }
    )


def test_sma_known_values():
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    result = indicators.sma(s, 3)
    assert np.isnan(result.iloc[1])
    assert result.iloc[2] == pytest.approx(2.0)
    assert result.iloc[4] == pytest.approx(4.0)


def test_ema_warmup_and_convergence():
    s = pd.Series([10.0] * 50)
    result = indicators.ema(s, 10)
    assert np.isnan(result.iloc[5])       # not defined during warm-up
    assert result.iloc[-1] == pytest.approx(10.0)  # converges on constant input


def test_zscore_symmetry():
    # Repeating pattern: mean 10, extremes equidistant -> symmetric z-scores.
    s = pd.Series([10.0, 12.0, 8.0] * 10)
    z = indicators.zscore(s, 6)
    assert z.dropna().max() == pytest.approx(-z.dropna().min(), rel=1e-9)


def test_zscore_flat_series_is_nan_not_inf():
    z = indicators.zscore(pd.Series([5.0] * 30), 10)
    assert not np.isinf(z.dropna()).any()


def test_true_range_includes_gaps(simple_df):
    tr = indicators.true_range(simple_df)
    # Bar 1: max(12-10.5, |12-10|, |10.5-10|) = 2.0
    assert tr.iloc[1] == pytest.approx(2.0)
    # Bar 0 has no previous close: plain high-low.
    assert tr.iloc[0] == pytest.approx(2.0)


def test_atr_positive_and_warmup(simple_df):
    df = pd.concat([simple_df] * 5, ignore_index=True)
    result = indicators.atr(df, 14)
    assert np.isnan(result.iloc[5])
    assert (result.dropna() > 0).all()


def test_rolling_high_excludes_current_bar():
    s = pd.Series([1.0, 2.0, 3.0, 10.0, 4.0])
    rh = indicators.rolling_high(s, 3)
    # At index 3 the prior-3 high is 3.0, so a close of 10 IS a breakout.
    assert rh.iloc[3] == pytest.approx(3.0)
    # At index 4 the spike is now in the lookback window.
    assert rh.iloc[4] == pytest.approx(10.0)
