from __future__ import print_function
import pandas as pd
import numpy as np
import pywt
import warnings
warnings.filterwarnings('ignore')

"""

HELPER:
Helper functions.

    1.  zero to nan
    2.  nan to zero
    3.  inf to nan 
    4.  refit mask
    5.  industry neutralize
    6.  apply booksize (CS)
    7.  positions to % (CS)
    8.  industry balancing (CS)
    9.  select n best positions (CS)
    10. even weighted portfolio
    11. moving average (TS)
    12. rolling std (TS)
    13. roll (TS)
    14. strides (numpy trick)
    15. rank values (CS)
    16. min-max (CS)
    17. zscore (CS)
    18. zscore min-max (CS)
    19. predictions weighting     
    20. prepare returns
    21. prepare returns (Z)
    22. wavelet transform

"""


# Fn: (1)
# Converts all zero and np.inf values to np.nan
def zero_to_nan(df):
    data = np.copy(df)
    data = np.array(data, dtype=np.float64)
    data[data == 0] = np.nan
    data[data == np.inf] = np.nan
    data[data == -np.inf] = np.nan
    return data


# Fn: (2)
# Converts all np.nan and np.inf values to zero
def nan_to_zero(df):
    data = np.copy(df)
    data = np.array(data, dtype=np.float64)
    data[np.isnan(data)] = 0
    data[data == np.inf] = 0
    data[data == -np.inf] = 0
    return data


# Fn: (3)
# Converts np.inf values to np.nan
def inf_to_nan(df):
    data = np.copy(df)
    data = np.array(data, dtype=np.float64)
    data[data == np.inf] = np.nan
    data[data == -np.inf] = np.nan
    return data


# Fn: (4)
# Arrange refit mask starting from the first fitting day.
def fn_refit_mask(dates, fit_startdate, refit_freq):
    refit_mask = np.zeros((dates.shape[0],), dtype=np.bool)
    refit_first_index = np.sort(np.where(dates > fit_startdate))[0][0]
    mask_arr = np.arange(refit_first_index, dates.shape[0], refit_freq + 1)
    refit_mask[mask_arr] = True
    return refit_mask


# Fn: (5)
# Neutralize matrix values to sector/industry/industrySubgroup effect.
def fn_industry_neutralization(df, industry_arr):
    data = np.copy(df)
    data = zero_to_nan(data)
    unique_industries = list(set(industry_arr))
    for industry in unique_industries:
        industry_idx = (industry_arr == industry)
        industry_data = np.copy(data[industry_idx, :])
        data[industry_idx, :] = industry_data - np.nanmean(industry_data, axis=0)
    return data


# Fn: (6)
# Share capital equally across long and short positions.
def fn_booksize(df, booksize=1e6, pos_limit=None):
    data = np.copy(df)
    # LONG
    # Share capital across long positions.
    longs_mask = (data > 0)
    longs_values = zero_to_nan(data * longs_mask)
    longs_pct = longs_values / np.nansum(longs_values, axis=0)
    longs_pos = nan_to_zero(longs_pct)
    # Check position limit if specified.
    if pos_limit is not None:
        long_max_mask = (longs_pos >= pos_limit)
        longs_exal = np.nansum(longs_pos[long_max_mask] - pos_limit, axis=0)
        longs_pos[~long_max_mask] += longs_pos[~long_max_mask] / \
                                     np.nansum(longs_pos[~long_max_mask], axis=0) * longs_exal
        longs_pos[long_max_mask] = pos_limit
        longs_pos = nan_to_zero(longs_pos)
        assert np.any(np.sum(longs_pos, axis=0) <= 1.00), 'allocation is wrong > abs(1.00)'
    # SHORT
    # Share capital across short positions.
    shorts_mask = (data < 0)
    shorts_values = zero_to_nan(data * shorts_mask)
    shorts_pct = shorts_values / np.nansum(shorts_values, axis=0)
    shorts_pos = nan_to_zero(shorts_pct * (-1.0))
    # Check position limit if specified.
    if pos_limit is not None:
        short_min_mask = (shorts_pos <= (pos_limit * (-1.0)))
        shorts_exal = np.nansum(shorts_pos[short_min_mask] + pos_limit, axis=0)
        shorts_pos[~short_min_mask] += shorts_pos[~short_min_mask] / \
                                       np.nansum(shorts_pos[~short_min_mask], axis=0) * shorts_exal
        shorts_pos[short_min_mask] = pos_limit * (-1.0)
        shorts_pos = nan_to_zero(shorts_pos)
        assert np.any(np.sum(longs_pos, axis=0) >= -1.00), 'allocation is wrong > abs(1.00)'
    # UNITED
    # Unite positions and allocate capital
    data = (longs_pos + shorts_pos) * booksize
    return data


# Fn: (7)
# Give percentage allocation values equally across long and short positions.
def fn_positions_go2pct(df):
    data = np.copy(df)
    # Balance across long positions.
    longs_mask = (data > 0)
    longs_values = zero_to_nan(data * longs_mask)
    longs_pct = longs_values / np.nansum(longs_values, axis=0)
    longs_pos = nan_to_zero(longs_pct)
    # Balance across short positions.
    shorts_mask = (data < 0)
    shorts_values = zero_to_nan(data * shorts_mask)
    shorts_pct = shorts_values / np.nansum(shorts_values, axis=0)
    shorts_pos = nan_to_zero(shorts_pct * (-1.0))
    # Unite positions.
    data = (longs_pos + shorts_pos)
    return data


# Fn: (8)
# Perform full cycle of industry balancing:
#   - Neutralize values
#   - Capital allocation
def fn_industry_balancing(df, industry_arr, booksize=1e6, pos_limit=None):
    data = np.copy(df)
    data = fn_industry_neutralization(data, industry_arr)
    data = fn_booksize(data, booksize=booksize, pos_limit=pos_limit)
    return data


# Fn: (9)
# Select only N best positions (via long and shorts positions separately).
def fn_select_n_best(df, n_best):
    data = np.copy(df)
    # Indices of best positions in short and long directions.
    indices_long = np.argpartition(data.transpose(), -n_best, axis=1)[:, -n_best:]
    indices_short = np.argpartition(data.transpose() * (-1.0), -n_best, axis=1)[:, -n_best:]
    for i in range(data.shape[1]):
        # Long positions.
        idx_long = indices_long[i]
        arr_long = np.copy(data[:, i])
        arr_long[list(set(np.arange(0, len(arr_long))) - set(idx_long))] *= 0
        # Short positions.
        idx_short = indices_short[i]
        arr_short = np.copy(data[:, i])
        arr_short[list(set(np.arange(0, len(arr_short))) - set(idx_short))] *= 0
        # Full portfolio
        data[:, i] = arr_long + arr_short
    return data


# Fn: (10)
# Create toy example of even weighted portfolio
def fn_positions_market_even(df, first_index):
    columns = df['dates'][first_index:].shape[0]
    rows = df['stocks'].shape[0]
    positions_market = np.zeros((rows, columns), dtype=np.float32)
    positions_market[:] = (1 / rows)
    return positions_market


# Fn: (11)
# Take mean value along time axis.
def ts_mean(df, window, hold_first=False):
    pre_data = np.copy(df)
    data = np.empty_like(pre_data)
    data[:] = np.nan
    data_strides = ts_strides(pre_data, window)
    for i in range(data_strides.shape[0]):
        data[i, window - 1:] = np.mean(data_strides[i], axis=1)
    if hold_first:
        data[:, :(window - 1)] = np.copy(df[:, :(window - 1)])
    return data


# Fn: (12)
# Get standard deviation value along time axis.
def ts_std(df, window):
    data = np.copy(df)
    data = pd.DataFrame(data.transpose())
    data = data.rolling(window=window, center=False).std()
    data = data.values.transpose()
    return data


# Fn: (13)
# Shift value along time axis.
def ts_delay(df, delay):
    data = np.copy(df)
    data = np.roll(data, delay, axis=1)
    if delay >= 0:
        data[:, :delay] = np.nan
    else:
        data[:, delay:] = np.nan
    return data


# Fn: (14)
# Strides building.
def ts_strides(x, window):
    shape = (x.shape[0], x.shape[1] - window + 1, window)
    strides = x.strides + (x.strides[-1],)
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)


# Fn: (15)
# Get rank values along stock data.
# Value range: [1, 2]
def cs_rank(df, max_zscore_cap=2.5):
    data = np.copy(df)
    data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
    data[data > max_zscore_cap] = max_zscore_cap
    data[data < -max_zscore_cap] = -max_zscore_cap
    data = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    data = data * (2.0 - 1.0) + 1.0
    return data


# Fn: (16)
# Range along stock data.
# Value range: [0.0, 1.0]
def cs_min_max(df):
    data = np.copy(df)
    data = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    return data


# Fn: (17)
# Range along stock data.
# Value range: [-zscore, zscore]
def cs_zscore(df, max_zscore_cap=4):
    data = np.copy(df)
    data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
    data[data > max_zscore_cap] = max_zscore_cap
    data[data < -max_zscore_cap] = -max_zscore_cap
    return data


# Fn: (18)
# Take zscore and range along stock data.
# Value range: [-1.0, 1.0]
def cs_zscore_min_max(df, max_zscore_cap=4, max_val=1, min_val=-1):
    data = np.copy(df)
    data = (data - np.nanmean(data, axis=0)) / np.nanstd(data, axis=0)
    data[data > max_zscore_cap] = max_zscore_cap
    data[data < -max_zscore_cap] = -max_zscore_cap
    data = (data - np.nanmin(data, axis=0)) / (np.nanmax(data, axis=0) - np.nanmin(data, axis=0))
    data = data * (max_val - min_val) + min_val
    return data


# Fn: (19)
# Handle multi-day predictions (2-rRNet or S2S-DeepConv-GRU) and return single vector:
def nn_handle_multi_day_predictions(input_prediction, ret_sizing, ret_adj):
    predictions = np.zeros((input_prediction.shape[0],), dtype=np.float32, order='F')
    day_1r = input_prediction[:, 0]
    day_2r = input_prediction[:, 1]
    day_3r = input_prediction[:, 2]
    total_ret = ((1 + day_1r) * (1 + day_2r) * (1 + day_3r) - 1)
    mask_pos = (day_1r + ret_sizing) < total_ret
    mask_neg = (day_1r - ret_sizing) > total_ret
    predictions[:] = day_1r
    predictions[mask_pos] = (day_1r[mask_pos] * (1 + ret_adj))
    predictions[mask_neg] = (day_1r[mask_neg] * (1 - ret_adj))
    return predictions


# Fn: (20)
# Prepare returns data (original)
def handle_returns(price, ndays, return_cap, min_target_keep, industry_balancing, industry):
    # Convert to returns.
    data = (price - ts_delay(price, ndays)) / ts_delay(price, ndays)
    data = np.array(data, dtype=np.float64)
    data = data / np.sqrt(ndays)
    # Cap return values.
    data[data > return_cap] = return_cap
    data[data < -return_cap] = -return_cap
    if industry_balancing:
        data = fn_industry_neutralization(data, industry)
    # Scale returns using zscore and eliminate small/big values.
    data = zero_to_nan(data)
    data[np.abs(data) < min_target_keep] = min_target_keep
    data = nan_to_zero(data)
    data = zero_to_nan(data)
    return data


# Fn: (21)
# Prepare returns data (CS:Zscore)
def handle_returns_z(price, ndays, return_cap, min_target_keep, industry_balancing, industry):
    # Convert to returns.
    data = (price - ts_delay(price, ndays)) / ts_delay(price, ndays)
    data = np.array(data, dtype=np.float64)
    data = data / np.sqrt(ndays)
    # Cap return values.
    data[data > return_cap] = return_cap
    data[data < -return_cap] = -return_cap
    if industry_balancing:
        data = fn_industry_balancing(data, industry, pos_limit=0.02)
    # Scale returns using zscore and eliminate small/big values.
    data = zero_to_nan(data)
    data = cs_zscore(data)
    data[np.abs(data) < min_target_keep] = min_target_keep
    data = nan_to_zero(data)
    data = zero_to_nan(data)
    return data


# Fn: (22)
# Build tensor via wavelet transform.
def get_wavelet(data, days, comps, q, p_id, wavelet_name='morl'):
    # Create tensor to fill.
    tensor = np.zeros((data.shape[0], data.shape[1] + days - 1, days, comps))
    tensor[:] = np.nan
    # Fill tensor with scalograms for every day on given stocks.
    for i in range(data.shape[0]):
        for j in range(data[i].shape[0]):
            coefficients, _ = pywt.cwt(data[i, j], range(1, comps + 1), wavelet_name)
            wavelet_scalogram = coefficients.transpose()
            tensor[i, days - 1 + j:, :, :] = wavelet_scalogram
    q.put((p_id, tensor))
