from __future__ import print_function
import numpy as np
import os
import h5py
import time
import math
import warnings
import configparser
import helper as hl
from scipy import stats
import multiprocessing as mp
from sklearn.externals import joblib
cpu_cores = mp.cpu_count()
warnings.filterwarnings('ignore')
np.random.seed(123)


class Alpha():

    """
    Create alphas.
    : input: data object.
    : return: -
    : store: save alphas to disk.

    Brief description:
        Derive alphas from given data.

    Alphas list:
        A: adv
        A: open p. momentum (1d)
        A: high p. momentum (1d)
        A: low p. momentum (1d)
        A: close p. momentum (1d)
        A: momentum (3d)
        A: momentum (10d)
        A: momentum (21d)
        A: momentum (70d)
        A: momentum (130d)
        A: sma (7d)
        A: sma (21d)
        A: macd
        A: bollinger bands
        A: standard deviation
        A: information ratio
        A: iqr
        A: 10% percentile
        A: 25% percentile
        A: 75% percentile
        A: 90% percentile
        A: positive cross count
        A: negative cross count
        A: median return
        A: mean return (3d)
        A: mean return (5d)
        A: mean return (10d)
        A: mean return (21d)
        A: mean reversion (3d)
        A: mean reversion (5d)
        A: mean reversion (10d)
        A: mean reversion (21d)
        A: dividends regular cash yield constant
        A: dividends regular cash yield relative
        A: dividends special cash yield constant
        A: dividends special cash yield relative
        A: eps earnings yield constant
        A: eps earnings yield relative
        A: eps earnings-estimate yield constant
        A: eps earnings-estimate yield relative
        A: eps earnings-comparable yield constant
        A: eps earnings-comparable yield relative
        E: dividends regular
        E: dividends special
        E: eps earnings
        E: eps comparable
        E: eps estimate
        T: original data
        T: wavelet transform

    """

    def __init__(self):

        # Path to options file.
        path_options = 'options.ini'

        # Load config file.
        config = configparser.ConfigParser()                                                # define config object.
        config.optionxform = str                                                            # hold keys register
        config.read(path_options)                                                           # read config file.

        # Path variables.
        self.path_data = config['PATH'].get('DataObj')                                      # address: data object
        self.path_alphas = config['PATH'].get('Alphas')                                     # address: alphas
        self.path_events = config['PATH'].get('Events')                                     # address: events
        self.path_tensors = config['PATH'].get('Tensors')                                   # address: tensors

        # Load data
        self.data = joblib.load(self.path_data)                                             # data object
        self.industry = self.data['industry.sector']                                        # industry data. arr.

        # Store alpha / event names in data object
        self.data['alphas'] = list()                                                        # alpha names list

        # Alpha params.
        self.a_adv_period = config['ALPHA'].getint('AlphaADVPeriod')                        # adv period
        self.a_macd_long_period = config['ALPHA'].getint('AlphaMACDLongPeriod')             # macd long period
        self.a_macd_short_period = config['ALPHA'].getint('AlphaMACDShortPeriod')           # macd short period
        self.a_boll_period = config['ALPHA'].getint('AlphaBollBPeriod')                     # boll.b. period
        self.a_boll_std = config['ALPHA'].getint('AlphaBollBStd')                           # boll.b. std
        self.a_std_period = config['ALPHA'].getint('AlphaRetStdPeriod')                     # std period
        self.a_ir_period = config['ALPHA'].getint('AlphaRetIRPeriod')                       # ir period
        self.a_iqr_period = config['ALPHA'].getint('AlphaRetIQRPeriod')                     # iqr period
        self.a_10p_period = config['ALPHA'].getint('AlphaRet10PPeriod')                     # 10 percentile period
        self.a_25p_period = config['ALPHA'].getint('AlphaRet25PPeriod')                     # 25 percentile period
        self.a_75p_period = config['ALPHA'].getint('AlphaRet75PPeriod')                     # 75 percentile period
        self.a_90p_period = config['ALPHA'].getint('AlphaRet90PPeriod')                     # 90 percentile period
        self.a_median_period = config['ALPHA'].getint('AlphaRetMedianPeriod')               # median ret period
        self.a_positive_cross_period = config['ALPHA'].getint('AlphaPosCrossPeriod')        # pos. cross period
        self.a_negative_cross_period = config['ALPHA'].getint('AlphaNegCrossPeriod')        # neg. cross period
        self.eps_earn_len = config['ALPHA'].getint('AlphaEPSEarnLen')                       # eps earn. yl. length
        self.eps_other_len = config['ALPHA'].getint('AlphaEPSOtherLen')                     # eps other yl. length
        self.div_reg_len = config['ALPHA'].getint('AlphaDIVRegLen')                         # div reg. yl. length
        self.div_spc_len = config['ALPHA'].getint('AlphaDIVSpcLen')                         # div spc. yl. length

        # Tensor params (original) .
        self.t_or_timesteps = config['TENSOR'].getint('TensorOrTimesteps')                  # tensor or. timesteps
        self.t_or_ret_cap = config['TENSOR'].getfloat('TensorOrRetCap')                     # tensor or. ret. cap
        self.t_or_min_target = config['TENSOR'].getfloat('TensorOrMinTarget')               # tensor or. min target
        self.t_or_ind_balancing = config['TENSOR'].getboolean('TensorOrIndBalancing')       # tensor or. ind. balance

        # Tensor params (scalogram)
        self.t_sc_timesteps = config['TENSOR'].getint('TensorScTimesteps')                  # tensor sc. timesteps
        self.t_sc_comps = config['TENSOR'].getint('TensorScComponents')                     # tensor sc. components
        self.t_sc_ret_cap = config['TENSOR'].getfloat('TensorScRetCap')                     # tensor sc. ret. cap
        self.t_sc_min_target = config['TENSOR'].getfloat('TensorScMinTarget')               # tensor sc. min target
        self.t_sc_ind_balancing = config['TENSOR'].getboolean('TensorScIndBalancing')       # tensor sc. ind. balance

        # Params
        self.neutralize = config['ALPHA'].getboolean('AlphaIndustryNeutralize ')            # activate industry neut.

    # Fn: (.)
    # Main function used to create alphas.
    def run_alphas(self):

        # Create alpha data folders
        # if they don`t exist.
        if not os.path.exists(self.path_alphas):
            os.makedirs(self.path_alphas)
        if not os.path.exists(self.path_events):
            os.makedirs(self.path_events)
        if not os.path.exists(self.path_tensors):
            os.makedirs(self.path_tensors)

        # Clean alpha data holders
        for file in os.listdir(self.path_alphas):
            os.remove(self.path_alphas + file)
        for file in os.listdir(self.path_events):
            os.remove(self.path_events + file)
        for file in os.listdir(self.path_tensors):
            os.remove(self.path_tensors + file)

        print('')
        print('Create alphas|events|tensors:')

        # ALPHA
        # It`s a signal in range ~{-3, 3}
        # based on signal power and balanced across
        # stocks traded on each day.
        # -----------------------
        # OHLCV indicators.
        self.alpha_adv()
        self.alpha_open_mom1d()
        self.alpha_high_mom1d()
        self.alpha_low_mom1d()
        self.alpha_close_mom1d()
        # Technical indicators.
        self.alpha_sma7()
        self.alpha_sma21()
        self.alpha_macd()
        self.alpha_close_mom3d()
        self.alpha_close_mom10d()
        self.alpha_close_mom21d()
        self.alpha_close_mom70d()
        self.alpha_close_mom130d()
        self.alpha_close_ret_ir()
        self.alpha_bollinger_bands()
        self.alpha_close_ret_median()
        self.alpha_close_ret_mean_3d()
        self.alpha_close_ret_mean_5d()
        self.alpha_close_ret_mean_10d()
        self.alpha_close_ret_mean_21d()
        self.alpha_close_ret_mean_reversion_3d()
        self.alpha_close_ret_mean_reversion_5d()
        self.alpha_close_ret_mean_reversion_10d()
        self.alpha_close_ret_mean_reversion_21d()
        # Dist. basic statistics
        self.alpha_close_ret_std()
        self.alpha_close_ret_iqr()
        self.alpha_close_ret_90p()
        self.alpha_close_ret_75p()
        self.alpha_close_ret_25p()
        self.alpha_close_ret_10p()
        self.alpha_close_ret_positive_cross()
        self.alpha_close_ret_negative_cross()
        # Dividends data.
        self.alpha_div_regular_yield_const()
        self.alpha_div_regular_yield_relative()
        self.alpha_div_special_yield_const()
        self.alpha_div_special_yield_relative()
        # EPS data.
        self.alpha_eps_earnings_yield_const()
        self.alpha_eps_earnings_yield_relative()
        self.alpha_eps_estimate_diff_yield_const()
        self.alpha_eps_estimate_diff_yield_relative()
        self.alpha_eps_comparable_diff_yield_const()
        self.alpha_eps_comparable_diff_yield_relative()

        # EVENT
        # Event is binary value {0, 1} indicating
        # some event happened the day before.
        # -------------------------
        self.event_div_regular()
        self.event_div_special()
        self.event_eps_earnings()
        self.event_eps_estimate_diff()
        self.event_eps_comparable_diff()

        # TENSOR
        # Tensor is a time series data holder
        # of OHLCV signals.
        # -------------------------
        self.tensor_original()
        self.tensor_wavelet()

        # Write number of alphas to data object
        self.data['numalphas'] = len(self.data['alphas'])
        print('  alphas: {}'.format(self.data['numalphas']))

        # Save data object.
        joblib.dump(self.data, self.path_data)
        print('alphas completed')

    # Fn: (1)
    # ALPHA: adv
    def alpha_adv(self):
        alpha_name = 'adv'
        alpha = hl.ts_mean(self.data['close'] * self.data['volume'], self.a_adv_period)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (2)
    # ALPHA: open p. momentum (1d)
    def alpha_open_mom1d(self):
        alpha_name = 'open_momentum'
        price = self.data['open']
        alpha = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (3)
    # ALPHA: high p. momentum (1d)
    def alpha_high_mom1d(self):
        alpha_name = 'high_momentum'
        price = self.data['high']
        alpha = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (4)
    # ALPHA: low p. momentum (1d)
    def alpha_low_mom1d(self):
        alpha_name = 'low_momentum'
        price = self.data['low']
        alpha = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (5)
    # ALPHA: close p. momentum (1d)
    def alpha_close_mom1d(self):
        alpha_name = 'close_momentum'
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (6)
    # ALPHA: momentum (3d)
    def alpha_close_mom3d(self):
        days = 3
        alpha_name = 'close_mom3d'
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, days)) / hl.ts_delay(price, days)
        alpha = alpha / np.sqrt(days)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (7)
    # ALPHA: momentum (10d)
    def alpha_close_mom10d(self):
        days = 10
        alpha_name = 'close_mom10d'
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, days)) / hl.ts_delay(price, days)
        alpha = alpha / np.sqrt(days)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (8)
    # ALPHA: momentum (21d)
    def alpha_close_mom21d(self):
        days = 21
        alpha_name = 'close_mom21d'
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, days)) / hl.ts_delay(price, days)
        alpha = alpha / np.sqrt(days)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (9)
    # ALPHA: momentum (70d)
    def alpha_close_mom70d(self):
        days = 70
        alpha_name = 'close_mom70d'
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, days)) / hl.ts_delay(price, days)
        alpha = alpha / np.sqrt(days)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (10)
    # ALPHA: momentum (130d)
    def alpha_close_mom130d(self):
        days = 130
        alpha_name = 'close_mom130d'
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, days)) / hl.ts_delay(price, days)
        alpha = alpha / np.sqrt(days)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (11)
    # ALPHA: SMA (7d)
    def alpha_sma7(self):
        alpha_name = 'sma7'
        price = self.data['close']
        alpha = hl.ts_mean(price, 7)
        alpha = (price - alpha) / alpha
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (12)
    # ALPHA: SMA (21d)
    def alpha_sma21(self):
        alpha_name = 'sma21'
        price = self.data['close']
        alpha = hl.ts_mean(price, 21)
        alpha = (price - alpha) / alpha
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (13)
    # ALPHA: MACD
    def alpha_macd(self):
        alpha_name = 'macd'
        price = self.data['close']
        alpha = hl.ts_mean(price, self.a_macd_long_period) - hl.ts_mean(price, self.a_macd_short_period)
        alpha = hl.ts_mean(alpha, self.a_macd_short_period)
        alpha = alpha / price
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (14)
    # ALPHA: Bollinger bands
    def alpha_bollinger_bands(self):
        alpha_name = 'bollinger_bands'
        price = self.data['close']
        price_std = hl.ts_std(price, self.a_boll_period)
        bollinger_upper_band = hl.ts_mean(price, self.a_boll_period) + (price_std * self.a_boll_std)
        bollinger_lower_band = hl.ts_mean(price, self.a_boll_period) - (price_std * self.a_boll_std)
        alpha = (price - bollinger_lower_band) / (bollinger_upper_band - bollinger_lower_band)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (15)
    # ALPHA: Standard deviation
    def alpha_close_ret_std(self):
        alpha_name = 'close_ret_std'
        price = self.data['close']
        price = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        alpha = hl.ts_std(price, self.a_std_period)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (16)
    # ALPHA: Information ratio
    def alpha_close_ret_ir(self):
        alpha_name = 'close_ret_ir'
        days = self.a_ir_period
        price = self.data['close']
        alpha = (price - hl.ts_delay(price, days)) / hl.ts_delay(price, days)
        alpha = np.abs(alpha / hl.ts_std(alpha, days))
        alpha = alpha / np.sqrt(days)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (17)
    # ALPHA: IQR
    def alpha_close_ret_iqr(self):
        days = self.a_iqr_period
        alpha_name = 'close_ret_iqr'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = stats.iqr(alpha_strides[i], axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (18)
    # ALPHA: 10% percentile
    def alpha_close_ret_10p(self):
        days = self.a_10p_period
        alpha_name = 'close_ret_10p'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.percentile(alpha_strides[i], q=10, axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (19)
    # ALPHA: 25% percentile
    def alpha_close_ret_25p(self):
        days = self.a_25p_period
        alpha_name = 'close_ret_25p'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.percentile(alpha_strides[i], q=25, axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (20)
    # ALPHA: 75% percentile
    def alpha_close_ret_75p(self):
        days = self.a_75p_period
        alpha_name = 'close_ret_75p'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.percentile(alpha_strides[i], q=75, axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (21)
    # ALPHA: 90% percentile
    def alpha_close_ret_90p(self):
        days = self.a_90p_period
        alpha_name = 'close_ret_90p'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.percentile(alpha_strides[i], q=90, axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (22)
    # ALPHA: Positive cross count
    def alpha_close_ret_positive_cross(self):
        days = self.a_positive_cross_period
        alpha_name = 'close_ret_positive_cross'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.count_nonzero((alpha_strides[i] > 0), axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (23)
    # ALPHA: Negative cross count
    def alpha_close_ret_negative_cross(self):
        days = self.a_negative_cross_period
        alpha_name = 'close_ret_negative_cross'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.count_nonzero((alpha_strides[i] < 0), axis=1)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        alpha *= (-1.0)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (24)
    # ALPHA: Median return
    def alpha_close_ret_median(self):
        days = self.a_median_period
        alpha_name = 'close_ret_median'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.median(alpha_strides[i], axis=1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (25)
    # ALPHA: Mean return (3d)
    def alpha_close_ret_mean_3d(self):
        days = 3
        alpha_name = 'close_ret_mean_3d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (26)
    # ALPHA: Mean return (5d)
    def alpha_close_ret_mean_5d(self):
        days = 5
        alpha_name = 'close_ret_mean_5d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (27)
    # ALPHA: Mean return (10d)
    def alpha_close_ret_mean_10d(self):
        days = 10
        alpha_name = 'close_ret_mean_10d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (28)
    # ALPHA: Mean return (21d)
    def alpha_close_ret_mean_21d(self):
        days = 21
        alpha_name = 'close_ret_mean_21d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (29)
    # ALPHA: Mean reversion (3d)
    def alpha_close_ret_mean_reversion_3d(self):
        days = 3
        alpha_name = 'close_ret_mean_reversion_3d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1) * (-1.0)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (30)
    # ALPHA: Mean reversion (5d)
    def alpha_close_ret_mean_reversion_5d(self):
        days = 5
        alpha_name = 'close_ret_mean_reversion_5d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1) * (-1.0)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (31)
    # ALPHA: Mean reversion (10d)
    def alpha_close_ret_mean_reversion_10d(self):
        days = 10
        alpha_name = 'close_ret_mean_reversion_10d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1) * (-1.0)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (32)
    # ALPHA: Mean reversion (21d)
    def alpha_close_ret_mean_reversion_21d(self):
        days = 21
        alpha_name = 'close_ret_mean_reversion_21d'
        price = self.data['close']
        alpha = np.zeros_like(price)
        alpha[:] = np.nan
        price_ret = (price - hl.ts_delay(price, 1)) / hl.ts_delay(price, 1)
        price_ret = hl.inf_to_nan(price_ret)
        alpha_strides = hl.ts_strides(price_ret, days)
        for i in range(alpha_strides.shape[0]):
            alpha[i, days - 1:] = np.mean(alpha_strides[i], axis=1) * (-1.0)
        alpha = hl.inf_to_nan(alpha)
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=3)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (33)
    # ALPHA: Dividends regular cash yield constant
    def alpha_div_regular_yield_const(self):
        alpha_name = 'div_regular_yield_const'
        data = self.data['div.regular.event']
        data = data * self.data['adj.price']
        data = data / self.data['close']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.div_reg_len)
                alpha[i, start:end] = val
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (34)
    # ALPHA: Dividends regular cash yield relative
    def alpha_div_regular_yield_relative(self):
        alpha_name = 'div_regular_yield_relative'
        data = self.data['div.regular.event']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.div_reg_len)
                alpha[i, start:end] = val
        alpha = alpha * self.data['adj.price']
        alpha = alpha / self.data['close']
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (35)
    # ALPHA: Dividends special cash yield constant.
    def alpha_div_special_yield_const(self):
        alpha_name = 'div_special_yield_const'
        data = self.data['div.special.event']
        data = data * self.data['adj.price']
        data = data / self.data['close']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.div_spc_len)
                alpha[i, start:end] = val
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (36)
    # ALPHA: Dividends special cash yield relative
    def alpha_div_special_yield_relative(self):
        alpha_name = 'div_special_yield_relative'
        data = self.data['div.special.event']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.div_spc_len)
                alpha[i, start:end] = val
        alpha = alpha * self.data['adj.price']
        alpha = alpha / self.data['close']
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (37)
    # ALPHA: EPS earnings yield constant
    def alpha_eps_earnings_yield_const(self):
        alpha_name = 'eps_earnings_yield_const'
        data = self.data['eps.earnings.event']
        data = data * self.data['adj.price']
        data = data / self.data['close']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.eps_earn_len)
                alpha[i, start:end] = val
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (38)
    # ALPHA: EPS earnings yield relative
    def alpha_eps_earnings_yield_relative(self):
        alpha_name = 'eps_earnings_yield_relative'
        data = self.data['eps.earnings.event']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.eps_earn_len)
                alpha[i, start:end] = val
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = alpha * self.data['adj.price']
        alpha = alpha / self.data['close']
        alpha = np.log(hl.zero_to_nan(alpha))
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (39)
    # ALPHA: EPS earnings-estimate yield constant
    def alpha_eps_estimate_diff_yield_const(self):
        alpha_name = 'eps_estimate_diff_yield_const'
        earnings = self.data['eps.earnings.event']
        estimate = self.data['eps.estimate.event']
        data = hl.inf_to_nan(earnings - estimate)
        data = data * self.data['adj.price']
        data = data / self.data['close']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.eps_other_len)
                alpha[i, start:end] = val
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (40)
    # ALPHA: EPS earnings-estimate yield relative
    def alpha_eps_estimate_diff_yield_relative(self):
        alpha_name = 'eps_estimate_diff_yield_relative'
        earnings = self.data['eps.earnings.event']
        estimate = self.data['eps.estimate.event']
        data = hl.inf_to_nan(earnings - estimate)
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.eps_other_len)
                alpha[i, start:end] = val
        alpha = alpha * self.data['adj.price']
        alpha = alpha / self.data['close']
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (41)
    # ALPHA: EPS earnings-comparable yield constant
    def alpha_eps_comparable_diff_yield_const(self):
        alpha_name = 'eps_comparable_diff_yield_const'
        earnings = self.data['eps.earnings.event']
        estimate = self.data['eps.comparable.event']
        data = hl.inf_to_nan(earnings - estimate)
        data = data * self.data['adj.price']
        data = data / self.data['close']
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.eps_other_len)
                alpha[i, start:end] = val
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (42)
    # ALPHA: EPS earnings-comparable yield relative
    def alpha_eps_comparable_diff_yield_relative(self):
        alpha_name = 'eps_comparable_diff_yield_relative'
        earnings = self.data['eps.earnings.event']
        estimate = self.data['eps.comparable.event']
        data = hl.inf_to_nan(earnings - estimate)
        alpha = np.empty_like(data, dtype=np.float32)
        alpha[:] = np.nan
        for i in range(data.shape[0]):
            f_idx = np.where(~np.isnan(data[i, :]))[0]
            f_val = data[i, f_idx]
            for ii, val in enumerate(f_val):
                start = f_idx[ii]
                end = np.add(start, self.eps_other_len)
                alpha[i, start:end] = val
        alpha = alpha * self.data['adj.price']
        alpha = alpha / self.data['close']
        if self.neutralize:
            alpha = hl.fn_industry_neutralization(alpha, self.industry)
        alpha = hl.cs_zscore(alpha, max_zscore_cap=2.5)
        joblib.dump(alpha, self.path_alphas + alpha_name + '.pickle')
        self.data['alphas'].append(alpha_name)

    # Fn: (43)
    # EVENT: Dividends regular
    def event_div_regular(self):
        event_name = 'div_regular'
        regular_cash = self.data['div.regular.event']
        event = (~np.isnan(regular_cash)) * 1.0
        joblib.dump(event, self.path_events + event_name + '.pickle')
        self.data['alphas'].append(event_name)

    # Fn: (44)
    # EVENT: Dividends special
    def event_div_special(self):
        event_name = 'div_special'
        special_cash = self.data['div.special.event']
        event = (~np.isnan(special_cash)) * 1.0
        joblib.dump(event, self.path_events + event_name + '.pickle')
        self.data['alphas'].append(event_name)

    # Fn: (45)
    # EVENT: EPS earnings
    def event_eps_earnings(self):
        event_name = 'eps_earnings'
        earnings = self.data['eps.earnings.event']
        event_pos = (earnings > 0) * 1.0
        event_neg = (earnings < 0) * (-1.0)
        event = event_pos + event_neg
        joblib.dump(event, self.path_events + event_name + '.pickle')
        self.data['alphas'].append(event_name)

    # Fn: (46)
    # EVENT: EPS comparable
    def event_eps_comparable_diff(self):
        event_name = 'eps_comparable_diff'
        earnings = self.data['eps.earnings.event']
        comparable = self.data['eps.comparable.event']
        event_pos = ((earnings - comparable) > 0) * 1.0
        event_neg = ((earnings - comparable) < 0) * (-1.0)
        event = event_pos + event_neg
        joblib.dump(event, self.path_events + event_name + '.pickle')
        self.data['alphas'].append(event_name)

    # Fn: (47)
    # EVENT: EPS estimate
    def event_eps_estimate_diff(self):
        event_name = 'eps_estimate_diff'
        earnings = self.data['eps.earnings.event']
        estimate = self.data['eps.estimate.event']
        event_pos = ((earnings - estimate) > 0) * 1.0
        event_neg = ((earnings - estimate) < 0) * (-1.0)
        event = event_pos + event_neg
        joblib.dump(event, self.path_events + event_name + '.pickle')
        self.data['alphas'].append(event_name)

    # Fn: (48)
    # TENSOR: Original data
    def tensor_original(self):
        fields = ['open', 'high', 'low', 'close']
        period = self.t_or_timesteps
        return_cap = self.t_or_ret_cap
        min_target = self.t_or_min_target
        industry_balancing = self.t_or_ind_balancing
        for field in fields:
            tensor_name = field + '_original'
            price = self.data[field]
            tensor = np.zeros((price.shape[0], price.shape[1], period))
            tensor[:] = np.nan
            ret_mat = hl.handle_returns(price, 1, return_cap, min_target, industry_balancing, self.industry)
            tensor_strides = hl.ts_strides(ret_mat, period)
            for i in range(tensor_strides.shape[0]):
                tensor[i, period - 1:, :] = tensor_strides[i]
            tensor = hl.zero_to_nan(tensor)
            tensor = hl.nan_to_zero(tensor)
            with h5py.File(self.path_tensors + tensor_name + '.h5', 'w') as hf:
                hf.create_dataset(tensor_name, data=tensor)

    # Fn: (49)
    # TENSOR: Wavelet transform
    def tensor_wavelet(self):
        fields = ['open', 'high', 'low', 'close', 'adv']
        period = self.t_sc_timesteps
        comps = self.t_sc_comps
        return_cap = self.t_sc_ret_cap
        min_target_keep = self.t_sc_min_target
        industry_balancing = True
        x_mat = None
        for field in fields[:]:
            tensor_name = field + '_wavelet'
            print('  building tensor: ' + field)
            if field != 'adv':
                price = self.data[field]
                x_mat = hl.handle_returns_z(price, 1, return_cap, min_target_keep, industry_balancing, self.industry)
            elif field == 'adv':
                x_mat = hl.ts_mean(self.data['close'] * self.data['volume'], 21)
                x_mat = hl.cs_zscore(hl.zero_to_nan(x_mat))
            strides = hl.ts_strides(x_mat, period)

            # MULTI-PROCESSING
            # Wavelet transform (CWT) requires heavy computations and we
            # will use map reduce to parallel that process on multiple cores.
            # ----------------------------------------------------------------
            q = mp.Queue()
            step = int(math.ceil(strides.shape[0] / cpu_cores))
            p_idx = np.arange(0, strides.shape[0] + step, step)
            # Define processes
            processes = [mp.Process(target=hl.get_wavelet, args=(strides[p_idx[c]:p_idx[c+1]], period, comps, q, c))
                         for c in range(cpu_cores)]
            # Start separate processes.
            for p in processes:
                p.start()
            # Get result from a queue.
            map_reduce_res = [q.get() for p in processes]
            # Terminate all processes.
            for p in processes:
                p.terminate()
            # Preprocess results
            map_reduce_res.sort()
            map_reduce_res = [r[1] for r in map_reduce_res]
            map_reduce_res = np.concatenate(map_reduce_res, axis=0)
            # Reassign tensor variable.
            tensor = map_reduce_res
            tensor = hl.zero_to_nan(tensor)
            tensor = hl.nan_to_zero(tensor)
            print('    tensor done')
            with h5py.File(self.path_tensors + tensor_name + '.h5', 'w') as hf:
                hf.create_dataset(tensor_name, data=tensor)

if __name__ == '__main__':

    t = time.time()
    Alpha().run_alphas()
    print(time.time() - t)
