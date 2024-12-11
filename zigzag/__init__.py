import pandas as pd
import numpy as np
from typing import Union

import zigzag_cython

# ATR related functions
def rma(series: pd.Series, length: int):
    length = int(length) if length and length > 0 else 10
    alpha = (1.0 / length) if length > 0 else 0.5

    # Calculate Result
    rma = series.ewm(alpha=alpha, min_periods=length).mean()
    return rma

def true_range(high: pd.Series, low: pd.Series, close: pd.Series):
    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    # ignore first row altough tr1 is not nan
    tr.iloc[0] = np.nan
    return tr

def average_true_range(high: pd.Series, low: pd.Series, close: pd.Series, length):
    _tr = true_range(high, low, close)
    return rma(_tr, length)


class ZigZag:
    def __init__(self, allow_zigzag_on_one_bar: bool = True):
        self.ALLOW_ZIGZAG_ON_ONE_BAR = allow_zigzag_on_one_bar

    def get_zigzag(
        self,
        high: Union[pd.Series, None] = None,
        low: Union[pd.Series, None] = None,
        close: Union[pd.Series, None] = None,
        candles: Union[pd.DataFrame, None] = None,
        min_dev_percent: float = 5,
        depth: int = 10,
    ) -> pd.DataFrame:
        """Trading View's ZigZag indicator implementation.

        Args:
            high (Union[pd.Series, None], optional): high series from OHLC. Defaults to None.
            low (Union[pd.Series, None], optional): low series from OHLC. Defaults to None.
            close (Union[pd.Series, None], optional): close series from OHLC. Defaults to None.
            candles (Union[pd.DataFrame, None], optional): A pandas DataFrame with columns ['high', 'low', 'close']. Defaults to None.
            min_dev float: The minimum price change to define a peak or a valley. Defaults to 5.
            depth int: The depth of the zigzag. Defaults to 10.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['pivot', 'pivot_confirmed_at'].
            pivot_kind: -1 for valley, 1 for peak, 0 for no pivot
        """

        if candles is not None:
            df = candles.copy()
            try:
                high = df["high"]
                low = df["low"]
                close = df["close"]
            except KeyError as e:
                raise KeyError(
                    "candles must have columns ['high', 'low', 'close']"
                ) from e
        else:
            df = pd.DataFrame({"high": high, "low": low, "close": close})
            # if high or low are not provided, generate from close
            df.high = df.max(axis=1)
            df.low = df.min(axis=1)

        df.dropna(inplace=True)

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()

        pivot, confirmed_idx = zigzag_cython.peak_valley_pivots(
            high,
            low,
            min_dev_percent,
            depth,
            allowed_zigzag_on_one_bar=self.ALLOW_ZIGZAG_ON_ONE_BAR,
        )
        pivot: np.ndarray = pivot
        confirmed_idx: np.ndarray = confirmed_idx

        # df["pivot"] should be -1, 0, 1
        # df["pivot_confirmed_at"] should be the timestamp of the confirmed pivot, else None

        df["pivot_kind"] = pivot
        df["pivot_confirmed_at"] = None
        df.loc[
            confirmed_idx != 0, "pivot_confirmed_at"
        ] = df.index[confirmed_idx[confirmed_idx != 0]]
        # if pivot is 0, change pivot_confirmed_at to None

        res = df[["pivot_kind", "pivot_confirmed_at"]].fillna(np.nan)

        return res


    def get_atr_zigzag(
            self,
            high: Union[pd.Series, None] = None,
            low: Union[pd.Series, None] = None,
            close: Union[pd.Series, None] = None,
            candles: Union[pd.DataFrame, None] = None,
            atr_len: int = 14,
            vol_amp: float = 3,
            min_dev:float = 5,
            max_dev:float = 15,
            depth:int = 10,
            min_abs_correction_size:float = 0,
            rel_edge_correction:float = 0,
        ) -> pd.DataFrame:
        """_summary_

        Args:
            high (Union[pd.Series, None], optional): high series from OHLC. Defaults to None.
            low (Union[pd.Series, None], optional): low series from OHLC. Defaults to None.
            close (Union[pd.Series, None], optional): close series from OHLC. Defaults to None.
            candles (Union[pd.DataFrame, None], optional): A pandas DataFrame with columns ['high', 'low', 'close']. Defaults to None.
            atr_len int: ATR length. Defaults to 14.
            vol_amp float: Volatility amplification factor. Defaults to 3.
            min_dev float: The minimum price change to define a peak or a valley. Defaults to 5.
            max_dev float: The maximum price change to define a peak or a valley. Defaults to 15.
            depth int: The depth of the zigzag. Defaults to 14.
            min_abs_correction_size (float, optional): Defaults to 0.
            rel_edge_correction (float, optional): Defaults to 0.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['pivot_kind', 'pivot_confirmed_at'].
            pivot_kind: -1 for valley, 1 for peak, 0 for no pivot
        """


        if candles is not None:
            df = candles.copy()
            try:
                high = df["high"]
                low = df["low"]
                close = df["close"]
            except KeyError as e:
                raise KeyError(
                    "candles must have columns ['high', 'low', 'close']"
                ) from e
        else:
            df = pd.DataFrame({"high": high, "low": low, "close": close})
            # if high or low are not provided, generate from close
            df.high = df.max(axis=1)
            df.low = df.min(axis=1)

        df["avg_vol"] = average_true_range(
            high=df["high"], low=df["low"], close=df["close"], length=atr_len, append=True
        )

        df.dropna(inplace=True)

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        atr = df["avg_vol"].to_numpy()

        pivot, confirmed_idx = zigzag_cython.atr_peak_valley_pivots(
            high,
            low,
            close,
            atr,
            vol_amp,
            min_dev,
            max_dev,
            rel_edge_correction,
            min_abs_correction_size,
            depth,
            allowed_zigzag_on_one_bar=self.ALLOW_ZIGZAG_ON_ONE_BAR,
        )
        pivot: np.ndarray = pivot
        confirmed_idx: np.ndarray = confirmed_idx

        # df["pivot"] should be -1, 0, 1
        # df["pivot_confirmed_at"] should be the timestamp of the confirmed pivot, else None

        df["pivot_kind"] = pivot
        df["pivot_confirmed_at"] = None
        df.loc[
            confirmed_idx != 0, "pivot_confirmed_at"
        ] = df.index[confirmed_idx[confirmed_idx != 0]]
        # if pivot is 0, change pivot_confirmed_at to None

        res = df[["pivot_kind", "pivot_confirmed_at"]].fillna(np.nan)

        return res