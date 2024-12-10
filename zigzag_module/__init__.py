import pandas as pd
import numpy as np
import pandas_ta as ta
from typing import Union

import zigzag

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

        pivot, confirmed_idx = zigzag.peak_valley_pivots(
            high,
            low,
            min_dev_percent / 100,
            depth,
            allowed_zigzag_on_one_bar=self.ALLOW_ZIGZAG_ON_ONE_BAR,
        )
        pivot: np.ndarray = pivot
        confirmed_idx: np.ndarray = confirmed_idx

        # df["pivot"] should be -1, 0, 1
        # df["pivot_confirmed_at"] should be the timestamp of the confirmed pivot, else None

        df["pivot"] = pivot
        df["pivot_confirmed_at"] = None
        df.loc[
            confirmed_idx != 0, "pivot_confirmed_at"
        ] = df.index[confirmed_idx[confirmed_idx != 0]]
        # if pivot is 0, change pivot_confirmed_at to None

        res = df[["pivot", "pivot_confirmed_at"]]

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
            min_abs_correction_size (float, optional): _description_. Defaults to 0.
            rel_edge_correction (float, optional): _description_. Defaults to 0.

        Returns:
            pd.DataFrame: A pandas DataFrame with columns ['pivot', 'pivot_confirmed_at'].
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

        df["avg_vol"] = ta.atr(
            high=df["high"], low=df["low"], close=df["close"], length=atr_len, append=True
        )

        df.dropna(inplace=True)

        high = df["high"].to_numpy()
        low = df["low"].to_numpy()
        close = df["close"].to_numpy()
        atr = df["avg_vol"].to_numpy()

        pivot, confirmed_idx = zigzag.atr_peak_valley_pivots(
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

        df["pivot"] = pivot
        df["pivot_confirmed_at"] = None
        df.loc[
            confirmed_idx != 0, "pivot_confirmed_at"
        ] = df.index[confirmed_idx[confirmed_idx != 0]]
        # if pivot is 0, change pivot_confirmed_at to None

        res = df[["pivot", "pivot_confirmed_at"]]

        return res

    def _generate_pivot(self, high_window: pd.Series, df: pd.DataFrame, depth: int):
        low_window = df["low"].loc[high_window.index]
        current_index = high_window.index[-1]
        # print(f'current_index: {current_index}')
        previous_index = high_window.index[-2]
        if "last_pivot" in df.columns:
            df.loc[current_index, "last_pivot"] = df.loc[previous_index, "last_pivot"]
            df.loc[current_index, "last_pivot_direction"] = df.loc[
                previous_index, "last_pivot_direction"
            ]
            df.loc[current_index, "last_pivot_price"] = df.loc[
                previous_index, "last_pivot_price"
            ]
            df.loc[current_index, "last_pivot_confirmed"] = df.loc[
                previous_index, "last_pivot_confirmed"
            ]

        peak_candidate = True
        if depth != 0:
            peak_candidate = True
            for i in range(0, depth):
                if high_window.iloc[i] >= high_window.iloc[depth]:
                    peak_candidate = False
                    break
            for i in range(depth + 1, depth * 2 + 1):
                if high_window.iloc[i] > high_window.iloc[depth]:
                    peak_candidate = False
                    break
        if peak_candidate:
            # as peak found in high_series[depth].index,
            if "last_pivot" not in df.columns:
                df["last_pivot"] = high_window.index[depth]
                df["last_pivot_direction"] = 1
                df["last_pivot_price"] = high_window.iloc[depth]
                df["last_pivot_confirmed"] = False
            else:
                last_pivot_direction = df["last_pivot_direction"].loc[current_index]
                last_pivot_price = df["last_pivot_price"].loc[current_index]

                if last_pivot_direction == 1:
                    # if last pivot was peak and current is also peak, update the last pivot
                    if high_window.iloc[depth] > last_pivot_price:
                        df.loc[current_index, "last_pivot"] = high_window.index[depth]
                        df.loc[current_index, "last_pivot_price"] = high_window.iloc[
                            depth
                        ]
                        df.loc[current_index, "last_pivot_confirmed"] = False
                else:
                    deviation_rate = (
                        (high_window.iloc[depth] - last_pivot_price)
                        / last_pivot_price
                        * 100
                    )
                    if deviation_rate >= df["dev_threshold"].loc[current_index]:
                        df.loc[current_index, "last_pivot"] = high_window.index[depth]
                        df.loc[current_index, "last_pivot_price"] = high_window.iloc[
                            depth
                        ]
                        df.loc[current_index, "last_pivot_direction"] = 1
                        df.loc[current_index, "last_pivot_confirmed"] = False

        valley_candidate = False
        if depth != 0:
            valley_candidate = True
            for i in range(0, depth):
                if low_window.iloc[i] <= low_window.iloc[depth]:
                    valley_candidate = False
                    break
            for i in range(depth + 1, depth * 2 + 1):
                if low_window.iloc[i] < low_window.iloc[depth]:
                    valley_candidate = False
                    break
        if (
            "last_pivot" in df.columns
            and df["last_pivot"].loc[current_index] == high_window.index[depth]
            and not self.ALLOW_ZIGZAG_ON_ONE_BAR
        ):
            valley_candidate = False
        if valley_candidate:
            if "last_pivot" not in df.columns:
                df["last_pivot"] = high_window.index[depth]
                df["last_pivot_direction"] = -1
                df["last_pivot_price"] = low_window[depth]
                df["last_pivot_confirmed"] = False
            else:
                last_pivot_direction = df["last_pivot_direction"].loc[current_index]
                last_pivot_price = df["last_pivot_price"].loc[current_index]

                if last_pivot_direction == -1:
                    # if last pivot was valley and current is also valley, update the last pivot
                    if low_window.iloc[depth] < last_pivot_price:
                        df.loc[current_index, "last_pivot"] = high_window.index[depth]
                        df.loc[current_index, "last_pivot_price"] = low_window.iloc[
                            depth
                        ]
                        df.loc[current_index, "last_pivot_confirmed"] = False
                else:
                    deviation_rate = (
                        (last_pivot_price - low_window.iloc[depth])
                        / last_pivot_price
                        * 100
                    )
                    if deviation_rate >= df["dev_threshold"].loc[current_index]:
                        df.loc[current_index, "last_pivot"] = high_window.index[depth]
                        df.loc[current_index, "last_pivot_price"] = low_window.iloc[
                            depth
                        ]
                        df.loc[current_index, "last_pivot_direction"] = -1
                        df.loc[current_index, "last_pivot_confirmed"] = False

        if (
            "last_pivot" in df.columns
            and not df["last_pivot_confirmed"].loc[current_index]
        ):
            last_pivot = df["last_pivot"].loc[current_index]
            # print(f'targeting {last_pivot}, direction: {df["last_pivot_direction"].loc[current_index]}')
            if df["last_pivot_direction"].loc[current_index] == 1:
                for i in df.loc[last_pivot:current_index].index[1:]:
                    # print(f"i: {i}")
                    confirm_criteria = df['edge_confirm_correction'].loc[i]
                    correction = (
                        # changed formula from the original idea
                        (df["last_pivot_price"].loc[current_index] - df["low"].loc[i]) / df["last_pivot_price"].loc[current_index]
                    ) * 100
                    # print(f"peak correction: {correction}, edge_confirm_correction: {edge_confirm_correction}")
                    if correction >= confirm_criteria:
                        df.loc[last_pivot, "pivot"] = 1.0
                        df.loc[current_index, "last_pivot_confirmed"] = True
                        df.loc[last_pivot, "pivot_confirmed_at"] = current_index
                        break
            else:
                for i in df.loc[last_pivot:current_index].index[1:]:
                    # print(f"i: {i}")
                    confirm_criteria = df['edge_confirm_correction'].loc[i]
                    correction = (
                        (df.loc[i, "high"] - df.loc[current_index, "last_pivot_price"])/ df.loc[current_index, "last_pivot_price"]
                    ) * 100
                    # print(f"valley correction: {correction}, edge_confirm_correction: {edge_confirm_correction}")
                    if correction >= confirm_criteria:
                        df.loc[last_pivot, "pivot"] = -1.0
                        df.loc[current_index, "last_pivot_confirmed"] = True
                        df.loc[last_pivot, "pivot_confirmed_at"] = current_index
                        break
        return 0.0