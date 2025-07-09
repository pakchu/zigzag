cimport cython

import numpy as np

from numpy cimport double_t, int64_t, ndarray

DEF PEAK = 1
DEF VALLEY = -1


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int64_t identify_initial_pivot(double [:] X,
                                   double up_thresh,
                                   double down_thresh):
    cdef:
        double x_0 = X[0]
        double x_t = x_0

        double max_x = x_0
        double min_x = x_0

        int64_t max_t = 0
        int64_t min_t = 0

    up_thresh += 1
    down_thresh += 1

    for t in range(1, len(X)):
        x_t = X[t]

        if x_t / min_x >= up_thresh:
            return VALLEY if min_t == 0 else PEAK

        if x_t / max_x <= down_thresh:
            return PEAK if max_t == 0 else VALLEY

        if x_t > max_x:
            max_x = x_t
            max_t = t

        if x_t < min_x:
            min_x = x_t
            min_t = t

    t_n = len(X)-1
    return VALLEY if x_0 < X[t_n] else PEAK

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

    res, confirmed_idx = peak_valley_pivots_detailed(
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
        allowed_zigzag_on_one_bar,
    )

    return res, confirmed_idx


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef peak_valley_pivots_detailed(double [:] HIGH,
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

    for t in range(depth * 2, t_n):
        high = HIGH[t]
        low = LOW[t]
        close = CLOSE[t]
        atr = ATR[t]

        raw_dev = atr / close * vol_amp * 100
        # clip with min_dev and max_dev
        clamped_vol = max(min_dev, min(max_dev, raw_dev))
        dev_threshold = clamped_vol
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
                    edge_confirm_correction = edge_confirm_corrections[t]
                    correction = last_pivot_price / LOW[i] - 1
                    if correction >= edge_confirm_correction:
                        pivots[last_pivot] = PEAK
                        last_pivot_confirmed = 1
                        pivot_confirmed_ats[last_pivot] = t
                        break
            else:
                for i in range(last_pivot + 1, t + 1):
                    edge_confirm_correction = edge_confirm_corrections[t]
                    correction = HIGH[i] / last_pivot_price - 1
                    if correction >= edge_confirm_correction:
                        pivots[last_pivot] = VALLEY
                        last_pivot_confirmed = 1
                        pivot_confirmed_ats[last_pivot] = t
                        break

    return pivots, pivot_confirmed_ats


def max_drawdown(X) -> float:
    X = _to_ndarray(X)

    # Ensure float for correct signature
    if not str(X.dtype).startswith('float'):
        X = X.astype(np.float64)

    return max_drawdown_c(X)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef double max_drawdown_c(ndarray[double, ndim=1] X):
    """
    Compute the maximum drawdown of some sequence.

    :return: 0 if the sequence is strictly increasing.
        otherwise the abs value of the maximum drawdown
        of sequence X
    """
    cdef:
        double mdd = 0
        double peak = X[0]
        double x, dd

    for x in X:
        if x > peak:
            peak = x

        dd = (peak - x) / peak

        if dd > mdd:
            mdd = dd

    return mdd if mdd != 0.0 else 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
def pivots_to_modes(int64_t [:] pivots):
    """
    Translate pivots into trend modes.

    :param pivots: the result of calling ``peak_valley_pivots``
    :return: numpy array of trend modes. That is, between (VALLEY, PEAK] it
    is 1 and between (PEAK, VALLEY] it is -1.
    """

    cdef:
        int64_t x, t
        ndarray[int64_t, ndim=1] modes = np.zeros(len(pivots),
                                                dtype=np.int64)
        int64_t mode = -pivots[0]

    modes[0] = pivots[0]

    for t in range(1, len(pivots)):
        x = pivots[t]
        if x != 0:
            modes[t] = mode
            mode = -x
        else:
            modes[t] = mode

    return modes


def compute_segment_returns(X, pivots):
    """
    :return: numpy array of the pivot-to-pivot returns for each segment."""
    X = _to_ndarray(X)
    pivot_points = X[pivots != 0]
    return pivot_points[1:] / pivot_points[:-1] - 1.0
