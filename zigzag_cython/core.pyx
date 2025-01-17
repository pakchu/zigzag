cimport cython
import numpy as np
from numpy cimport ndarray, int64_t, double_t

DEF PEAK = 1
DEF VALLEY = -1


def _to_ndarray(X):
    # The type signature in peak_valley_pivots_detailed does not work for
    # pandas series because as of 0.13.0 it no longer sub-classes ndarray.
    # The workaround everyone used was to call `.values` directly before
    # calling the function. Which is fine but a little annoying.
    t = type(X)
    if t.__name__ == 'ndarray':
        pass  # Check for ndarray first for historical reasons
    elif f"{t.__module__}.{t.__name__}" == 'pandas.core.series.Series':
        X = X.values
    elif isinstance(X, (list, tuple)):
        X = np.array(X)

    return X


def peak_valley_pivots(
    HIGH,
    LOW,
    cHIGH,
    cLOW,
    min_dev,
    depth,
    allowed_zigzag_on_one_bar
):
    HIGH = _to_ndarray(HIGH)
    LOW = _to_ndarray(LOW)
    cHIGH = _to_ndarray(cHIGH)
    cLOW = _to_ndarray(cLOW)

    # Ensure float for correct signature
    if not str(HIGH.dtype).startswith('float'):
        HIGH = HIGH.astype(np.float64)

    if not str(LOW.dtype).startswith('float'):
        LOW = LOW.astype(np.float64)

    res, confirmed_idx = peak_valley_pivots_detailed(
        HIGH,
        LOW,
        cHIGH,
        cLOW,
        min_dev,
        depth,
        allowed_zigzag_on_one_bar
    )

    return res, confirmed_idx


def atr_peak_valley_pivots(
    HIGH,
    LOW,
    CLOSE,
    ATR,
    vol_amp,
    min_dev,
    max_dev,
    rel_edge_correction,
    min_abs_correction_size,
    depth,
    allowed_zigzag_on_one_bar
):
    HIGH = _to_ndarray(HIGH)
    LOW = _to_ndarray(LOW)
    CLOSE = _to_ndarray(CLOSE)
    ATR = _to_ndarray(ATR)

    # Ensure float for correct signature
    if not str(HIGH.dtype).startswith('float'):
        HIGH = HIGH.astype(np.float64)

    if not str(LOW.dtype).startswith('float'):
        LOW = LOW.astype(np.float64)

    if not str(CLOSE.dtype).startswith('float'):
        CLOSE = CLOSE.astype(np.float64)

    if not str(ATR.dtype).startswith('float'):
        ATR = ATR.astype(np.float64)

    res, confirmed_idx = atr_peak_valley_pivots_detailed(
        HIGH,
        LOW,
        CLOSE,
        ATR,
        vol_amp,
        min_dev,
        max_dev,
        rel_edge_correction,
        min_abs_correction_size,
        depth,
        allowed_zigzag_on_one_bar
    )

    return res, confirmed_idx


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef peak_valley_pivots_detailed(double [:] HIGH,
                                  double [:] LOW,
                                  double [:] cHIGH,
                                  double [:] cLOW,
                                  double min_dev,
                                  int depth,
                                  int allowed_zigzag_on_one_bar):

    cdef:
        int64_t t_n = len(HIGH)
        ndarray[int64_t, ndim=1] pivots = np.zeros(t_n, dtype=np.int64)
        ndarray[int64_t, ndim=1] pivot_confirmed_ats = np.zeros(t_n, dtype=np.int64)
        int64_t last_pivot = -1
        int64_t last_pivot_direction
        double_t last_pivot_price
        int64_t last_pivot_confirmed
        int64_t peak_candidate
        int64_t valley_candidate
        # index types
        int64_t t
        int64_t current_pivot_target
        int64_t i
        double_t dev_threshold


    for t in range(depth * 2, t_n):
        dev_threshold = min_dev

        peak_candidate = 1
        current_pivot_target = t - depth
        current_pivot_high = HIGH[current_pivot_target]
        current_pivot_chigh = cHIGH[current_pivot_target]
        for i in range(current_pivot_target - depth, current_pivot_target):
            if HIGH[i] >= current_pivot_high:
                peak_candidate = 0
                break
        for i in range(current_pivot_target + 1, current_pivot_target + depth + 1):
            if HIGH[i] > current_pivot_high:
                peak_candidate = 0
                break
        if peak_candidate == 1:
            if last_pivot == -1:
                last_pivot = current_pivot_target
                last_pivot_direction = 1
                last_pivot_price = current_pivot_high
                last_pivot_confirmed = 0
            else:
                if last_pivot_direction == 1:
                    if current_pivot_high > last_pivot_price:
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_high
                        last_pivot_confirmed = 0
                else:
                    deviation_rate = (current_pivot_chigh - last_pivot_price) / last_pivot_price * 100
                    if deviation_rate >= dev_threshold:
                        pivots[last_pivot] = VALLEY
                        last_pivot_confirmed = 1
                        pivot_confirmed_ats[last_pivot] = t
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_high
                        last_pivot_direction = 1
                        last_pivot_confirmed = 0

        valley_candidate = 1
        current_pivot_target = t - depth
        current_pivot_low = LOW[current_pivot_target]
        current_pivot_clow = cLOW[current_pivot_target]
        for i in range(current_pivot_target - depth, current_pivot_target):
            if LOW[i] <= current_pivot_low:
                valley_candidate = 0
                break
        for i in range(current_pivot_target + 1, current_pivot_target + depth + 1):
            if LOW[i] < current_pivot_low:
                valley_candidate = 0
                break
        if last_pivot != -1 and last_pivot == current_pivot_target and allowed_zigzag_on_one_bar == 0:
            valley_candidate = 0
        if valley_candidate == 1:
            if last_pivot == -1:
                last_pivot = current_pivot_target
                last_pivot_direction = -1
                last_pivot_price = current_pivot_low
                last_pivot_confirmed = 0
            else:
                if last_pivot_direction == -1:
                    if current_pivot_low < last_pivot_price:
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_low
                        last_pivot_confirmed = 0
                else:
                    deviation_rate = (last_pivot_price - current_pivot_clow) / last_pivot_price * 100
                    if deviation_rate >= dev_threshold:
                        pivots[last_pivot] = PEAK
                        last_pivot_confirmed = 1
                        pivot_confirmed_ats[last_pivot] = t
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_low
                        last_pivot_direction = -1
                        last_pivot_confirmed = 0


    return pivots, pivot_confirmed_ats


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef atr_peak_valley_pivots_detailed(double [:] HIGH,
                                  double [:] LOW,
                                  double [:] CLOSE,
                                  double [:] ATR,
                                  double vol_amp,
                                  double min_dev,
                                  double max_dev,
                                  double rel_edge_correction,
                                  double min_abs_correction_size,
                                  int depth,
                                  int allowed_zigzag_on_one_bar):

    cdef:
        int64_t t_n = len(HIGH)
        ndarray[int64_t, ndim=1] pivots = np.zeros(t_n, dtype=np.int64)
        ndarray[double_t, ndim=1] edge_confirm_corrections = np.zeros(t_n, dtype=np.double)
        ndarray[int64_t, ndim=1] pivot_confirmed_ats = np.zeros(t_n, dtype=np.int64)
        int64_t last_pivot = -1
        int64_t last_pivot_direction
        double_t last_pivot_price
        int64_t last_pivot_confirmed
        int64_t peak_candidate
        int64_t valley_candidate
        # index types
        int64_t t
        int64_t current_pivot_target
        int64_t i
        double_t dev_threshold


    for t in range(depth * 2, t_n):
        raw_dev = ATR[t] / CLOSE[t] * vol_amp * 100
        # clip with min_dev and max_dev
        dev_threshold = max(min_dev, min(max_dev, raw_dev))
        edge_confirm_correction = dev_threshold * rel_edge_correction
        edge_confirm_correction = max(min_abs_correction_size * 100, edge_confirm_correction) / 100
        edge_confirm_corrections[t] = edge_confirm_correction

        peak_candidate = 1
        current_pivot_target = t - depth
        current_pivot_high = HIGH[current_pivot_target]
        for i in range(current_pivot_target - depth, current_pivot_target):
            if HIGH[i] >= current_pivot_high:
                peak_candidate = 0
                break
        for i in range(current_pivot_target + 1, current_pivot_target + depth + 1):
            if HIGH[i] > current_pivot_high:
                peak_candidate = 0
                break
        if peak_candidate == 1:
            if last_pivot == -1:
                last_pivot = current_pivot_target
                last_pivot_direction = 1
                last_pivot_price = current_pivot_high
                last_pivot_confirmed = 0
            else:
                if last_pivot_direction == 1:
                    if current_pivot_high > last_pivot_price:
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_high
                        last_pivot_confirmed = 0
                else:
                    deviation_rate = (current_pivot_high - last_pivot_price) / last_pivot_price * 100
                    if deviation_rate >= dev_threshold:
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_high
                        last_pivot_direction = 1
                        last_pivot_confirmed = 0

        valley_candidate = 1
        current_pivot_target = t - depth
        current_pivot_low = LOW[current_pivot_target]
        for i in range(current_pivot_target - depth, current_pivot_target):
            if LOW[i] <= current_pivot_low:
                valley_candidate = 0
                break
        for i in range(current_pivot_target + 1, current_pivot_target + depth + 1):
            if LOW[i] < current_pivot_low:
                valley_candidate = 0
                break
        if last_pivot != -1 and last_pivot == current_pivot_target and allowed_zigzag_on_one_bar == 0:
            valley_candidate = 0
        if valley_candidate == 1:
            if last_pivot == -1:
                last_pivot = current_pivot_target
                last_pivot_direction = -1
                last_pivot_price = current_pivot_low
                last_pivot_confirmed = 0
            else:
                if last_pivot_direction == -1:
                    if current_pivot_low < last_pivot_price:
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_low
                        last_pivot_confirmed = 0
                else:
                    deviation_rate = (last_pivot_price - current_pivot_low) / last_pivot_price * 100
                    if deviation_rate >= dev_threshold:
                        last_pivot = current_pivot_target
                        last_pivot_price = current_pivot_low
                        last_pivot_direction = -1
                        last_pivot_confirmed = 0

        if last_pivot != -1 and last_pivot_confirmed == 0:
            if last_pivot_direction == 1:
                for i in range(last_pivot + 1, t + 1):
                    edge_confirm_correction = edge_confirm_corrections[i]
                    correction = last_pivot_price / LOW[i] - 1
                    if correction >= edge_confirm_correction:
                        pivots[last_pivot] = PEAK
                        last_pivot_confirmed = 1
                        pivot_confirmed_ats[last_pivot] = t
                        break
            else:
                for i in range(last_pivot + 1, t + 1):
                    edge_confirm_correction = edge_confirm_corrections[i]
                    correction = HIGH[i] / last_pivot_price - 1
                    if correction >= edge_confirm_correction:
                        pivots[last_pivot] = VALLEY
                        last_pivot_confirmed = 1
                        pivot_confirmed_ats[last_pivot] = t
                        break

    return pivots, pivot_confirmed_ats
