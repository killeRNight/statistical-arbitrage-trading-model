from __future__ import print_function
import pandas as pd
import numpy as np
import os
import time
import math
import warnings
import helper as hl
import configparser
from sklearn.externals import joblib
import multiprocessing as mp
cpu_cores = mp.cpu_count()
np.random.seed(123)
warnings.filterwarnings('ignore')


class Data():

    """
    Convert raw data to required format.
    : input: given raw data.
    : return: -
    : store: save data object to disk.

    Brief description:
        Pre-process given raw data and converts it to data object
        containing data matrices of following format (stocks * days).

    Prepare data:
        1. Resample given data to 1-day format.
        2. Select only following stocks:
            - enough data for analysis.
            - maximum gap length is 1.
            - have information about corporate events.
            - company corporate events actual history do not include chosen events.
        3. Get EPS events.
        4. Get Dividend events.
        5. Get sector codes.
        6. Adjust data on stock splits and dividends.

    """

    def __init__(self):

        # Path to options file.
        path_options = 'options.ini'

        # Data object
        self.data = dict()                                                               # main data structure

        # Load config file.
        config = configparser.ConfigParser()                                             # define config object.
        config.optionxform = str                                                         # hold keys register
        config.read(path_options)                                                        # read config file.

        # Path variables.
        self.path_data = config['PATH'].get('DataObj')                                   # address: data object
        self.path_raw_15M_data = config['PATH'].get('Price15minData')                    # address: price 15M data
        self.path_raw_1D_data = config['PATH'].get('Price1dayData')                      # address: price 1D data
        self.path_sector_codes = config['PATH'].get('SectorCodes')                       # address: sector codes
        self.path_raw_div = config['PATH'].get('DividendsData')                          # address: dividends data
        self.path_raw_eps = config['PATH'].get('EpsData')                                # address: eps data

        # Params.
        self.stocks_arr = os.listdir(self.path_raw_15M_data)[:]                          # stocks list
        self.data_min_length = config['DATA'].getint('DataMinLength')                    # min days to analyze stock
        self.ohlcv_fields = ['open', 'high', 'low', 'close', 'volume']                   # stock data structure
        self.corp_events_ignore = ['Poison Pill Rights', 'In-specie',                    # corp event to ignore
                                   'Spinoff', 'Split-Off']                               # ..........

        # Create data folder is it doesn`t exist.
        if not os.path.exists(self.path_data.replace('data.pickle', '')):
            os.makedirs(self.path_data.replace('data.pickle', ''))

        # Delete old data object.
        try:
            os.remove(self.path_data)
        except FileNotFoundError:
            pass

    # Fn: (1)
    # Handle all necessary operations to convert
    # raw data to matrices.
    def prepare_data(self):

        print('Prepare data: ')
        print('  stocks: {} (start)'.format(len(self.stocks_arr)))

        # Stocks we'll not use for analysis.
        stocks_to_delete_idx = list()
        # Sort stocks.
        self.stocks_arr = np.array(list((map(lambda x: x.split('.csv')[0], self.stocks_arr))))
        self.stocks_arr = np.sort(np.array(self.stocks_arr))[:]

        # Clean data holders.
        for field in self.ohlcv_fields:
            for file in os.listdir(self.path_raw_1D_data + field + '/'):
                os.remove(self.path_raw_1D_data + field + '/' + file)

        # MAP REDUCE
        # Resample data using parallel processes on multiple cores
        # ---------------------------------------------------------
        q = mp.Queue()
        step = int(math.ceil(self.stocks_arr.shape[0] / cpu_cores))
        p_idx = np.arange(0, self.stocks_arr.shape[0] + step, step)
        # Define processes
        processes = [mp.Process(target=self.resample_data, args=(self.stocks_arr[p_idx[c]:p_idx[c + 1]], q, c))
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
        unique_dates = np.unique(map_reduce_res)
        # Data: convert dates to int.
        dates_df = pd.DataFrame(data={'dates': unique_dates})
        dates_df = dates_df.sort_values(by=['dates'])
        dates_df = dates_df.reset_index(drop=True)
        dates_all = dates_df['dates'].dt.strftime('%Y%m%d').values.astype(np.int32)

        """
        DATA: 
             OHLCV

        DESCRIPTION:    
            Our model works w/ data in following format:       
                - RAW: stocks 
                - COLUMN: dates               
            For every feature we have the same 2D matrix with all stocks united together.   

        EXAMPLE:
            So, if we have CLOSE stock price data of Apple and Facebook for 360 days,
            then our data matrix will have (2, 360) shape. Using such a data structure will
            speed up the development process.

        """
        # Go through each field: adjust and save OHLCV data.
        for field in self.ohlcv_fields:

            # Create empty 2d matrix to fill w/ stock data.
            df_ohlcv = np.zeros((self.stocks_arr.shape[0], dates_all.shape[0]), dtype=np.float32)
            df_ohlcv[:] = np.nan

            # Go through every stock to unite all data in one matrix.
            for stock_idx, stock in enumerate(self.stocks_arr):

                # Load stock data.
                df_stock = pd.read_csv(self.path_raw_1D_data + field + '/' + stock + '.csv', header=0)
                df_stock = df_stock.sort_values(by=['dt'])
                stock_ohlcv = df_stock[field].values

                # DATA LENGTH
                # Check if data is enough for analysis or not.
                # If not, drop that stock and continue.
                # -------------------------------------------------
                if stock_ohlcv.shape[0] <= self.data_min_length:
                    stocks_to_delete_idx.append(stock_idx)
                    continue

                # Write stock data to united matrix
                stock_dates = df_stock['dt'].values
                stock_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), stock_dates)))
                stock_dates_indices = np.where(np.in1d(dates_all, stock_dates))[0]
                df_ohlcv[stock_idx, stock_dates_indices] = stock_ohlcv

                # GAPS
                # Check gaps in given data. If data has more than 1
                # NaN value, we drop that stock.
                # -------------------------------------------------
                start = np.where(dates_all == stock_dates[0])[0][0]
                end = np.where(dates_all == stock_dates[-1])[0][0]
                stock_price_in_df_stock = df_ohlcv[stock_idx, start:(end + 1)]
                nan_sum = np.sum(np.isnan(stock_price_in_df_stock))
                # Fill missing values with given data.
                if nan_sum == 1:
                    # no change if it is the first value.
                    if stock_price_in_df_stock[0] == np.nan:
                        pass
                    # forward fill if nan value is last
                    elif stock_price_in_df_stock[-1] == np.nan:
                        stock_price_in_df_stock[-1] = stock_price_in_df_stock[-2]
                    # forward fill in nan value is in the middle
                    else:
                        nan_idx = np.argwhere(np.isnan(stock_price_in_df_stock))[0][0]
                        stock_price_in_df_stock[nan_idx] = stock_price_in_df_stock[nan_idx - 1]
                # Delete data and stock if NaN in total is more than 1
                elif nan_sum > 1:
                    stocks_to_delete_idx.append(stock_idx)
                    continue

                # CORPORATE EVENTS
                # Check if div. data has an information about certain
                # corporate event happened on dates stock traded.
                # If so, drop that stock.
                # -------------------------------------------------
                try:
                    df_div = pd.read_csv(self.path_raw_div + '/' + stock + '.csv', header=0)
                    corp_events = df_div['Dividend Type'].values
                    for c in self.corp_events_ignore:
                        c_event_dates = df_div['Ex-Date'].values[corp_events == c]
                        c_event_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), c_event_dates)))
                        c_idx = np.count_nonzero(np.in1d(c_event_dates, stock_dates))
                        if c_idx > 0:
                            stocks_to_delete_idx.append(stock_idx)

                # NO CORPORATE EVENTS
                # If there is no information about corporate events for
                # a given stock, we drop that stock due to high uncertainty.
                # --------------------------------------------------
                except FileNotFoundError:
                    stocks_to_delete_idx.append(stock_idx)

            # Save OHLCV to data dictionary.
            self.data[field] = df_ohlcv
        # -----------------------------------------------------------------------------------

        """
        # EPS
        # ------------------------------------------
        # Matrices containing information about EPS {Events & Trailing Quarter)
        # I decided to choice T4M method cause we rank every alpha across evey day, not every stock.

        """
        # Events containing in EPS data.
        eps_fields = ['Comparable EPS', 'Earnings EPS', 'Estimate EPS']
        for e_field in eps_fields:
            # Define data holder.
            df_eps_event = np.zeros((self.stocks_arr.shape[0], dates_all.shape[0]), dtype=np.float32)
            df_eps_event[:] = np.nan
            for stock_idx, stock in enumerate(self.stocks_arr[:]):
                # Skip if stock is irrelevant.
                if stock_idx in stocks_to_delete_idx:
                    continue

                # Load EPS data.
                df_eps = pd.read_csv(self.path_raw_eps + '/' + stock + '.csv', header=0)
                df_eps = df_eps.sort_values(by=['Announcement Date'], ascending=False)
                eps_event = df_eps[e_field].values
                eps_dates = df_eps['Announcement Date'].values
                eps_time = df_eps['Announcement Time']
                eps_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), eps_dates)))
                eps_quarter = df_eps['Year/Period'].values

                # NAN VALUES
                # ---------------------------
                # Set valid value.
                eps_valid_min = 0.0000001
                eps_valid_max = 1000000.00
                eps_event[np.abs(eps_event) < eps_valid_min] = 0
                eps_event[np.abs(eps_event) > eps_valid_max] = np.nan
                # Clean nan values
                eps_nan = np.copy(~np.isnan(eps_event))
                eps_event = eps_event[eps_nan]
                eps_dates = eps_dates[eps_nan]
                eps_quarter = eps_quarter[eps_nan]
                eps_time = eps_time[eps_nan]
                # Clean announcement time nan
                eps_nan = ~pd.isnull(eps_time).values
                eps_event = eps_event[eps_nan]
                eps_dates = eps_dates[eps_nan]
                eps_quarter = eps_quarter[eps_nan]

                # DUPLICATES
                # ----------------------------
                # Replace all duplicate values with the newest values.
                duplicates = eps_dates[np.setdiff1d(np.arange(len(eps_dates)),
                                                    np.unique(eps_dates, return_index=True)[1])]
                dup_idx_del = []
                # Go through each duplicate value.
                for dup in duplicates:
                    dup_idx = np.where(eps_dates == dup)[0]
                    dup_quarters = eps_quarter[dup_idx]
                    best_quarter = np.max(dup_quarters)
                    irrelevant_quarter_idx = np.where(dup_quarters != best_quarter)[0]
                    dup_idx_del.append(dup_idx[irrelevant_quarter_idx])
                # Delete duplicate values from EPS data holders
                if len(dup_idx_del) > 0:
                    dup_idx_del = np.concatenate(dup_idx_del).ravel()
                    eps_event = np.delete(eps_event, dup_idx_del, axis=0)
                    eps_dates = np.delete(eps_dates, dup_idx_del, axis=0)

                # Get correct indices
                eps_dates_indices = np.where(np.in1d(dates_all, eps_dates))[0]
                dates_eps_indices = np.where(np.in1d(eps_dates, dates_all))[0]
                # Update values
                eps_event = eps_event[dates_eps_indices][::-1]
                # Fill values to matrix.
                df_eps_event[stock_idx, eps_dates_indices] = eps_event

            # Save EPS to data dictionary.
            df_eps_event = hl.ts_delay(df_eps_event, 1)
            df_eps_event = hl.zero_to_nan(hl.nan_to_zero(df_eps_event))
            self.data['eps.' + e_field.lower().split(' ')[0] + '.event'] = df_eps_event
        # -----------------------------------------------------------------------------

        """
        # DIVIDENDS
        # ------------------------------------------
        # Matrices containing information about dividends {Events & Trailing Quarter).
        # Events: regular cash and special cash.

        """
        # Events we`re working with.
        div_fields = ['Regular Cash', 'Special Cash']
        for d_field in div_fields:
            # Define data holders.
            df_div_event = np.zeros((self.stocks_arr.shape[0], dates_all.shape[0]), dtype=np.float32)
            df_div_payday = np.zeros((self.stocks_arr.shape[0], dates_all.shape[0]), dtype=np.int32)
            df_div_event[:] = np.nan
            for stock_idx, stock in enumerate(self.stocks_arr[:]):
                # Skip if stock is irrelevant.
                if stock_idx in stocks_to_delete_idx:
                    continue

                # Load DIV data.
                df_div = pd.read_csv(self.path_raw_div + '/' + stock + '.csv', header=0)
                df_div = df_div.sort_values(by=['Ex-Date'], ascending=False)
                event_mask = (df_div['Dividend Type'].values == d_field)
                div_event = df_div['Dividend Amount'].values[event_mask]
                div_dates = df_div['Ex-Date'].values[event_mask]
                div_dates = np.array(list(map(lambda x: np.int32(x.replace('-', '')), div_dates)))

                try:
                    div_payday = df_div['Payable Date'].values[event_mask]
                    div_payday = np.array(list(map(lambda x: np.int32(x.replace('-', '')), div_payday)))
                except AttributeError:
                    div_payday = df_div['Payable Date'].fillna('0000-00-00').values[event_mask]
                    div_payday = np.array(list(map(lambda x: np.int32(x.replace('-', '')), div_payday)))

                # NAN VALUES
                # --------------------------------
                # Set valid value.
                div_valid_min = 0.0000001
                div_valid_max = 1000000.00
                div_event[np.abs(div_event) < div_valid_min] = 0
                div_event[np.abs(div_event) > div_valid_max] = np.nan
                div_nan = np.copy(~np.isnan(div_event))
                # Clean nan values
                div_event = div_event[div_nan]
                div_dates = div_dates[div_nan]
                div_payday = div_payday[div_nan]

                # DUPLICATES
                # ---------------------------------
                # Replace all duplicate values with the newest values.
                duplicates = div_dates[np.setdiff1d(np.arange(len(div_dates)),
                                                    np.unique(div_dates, return_index=True)[1])]
                dup_idx_del = []
                # Go through each duplicate value.
                for dup in duplicates:
                    dup_idx = np.where(div_dates == dup)[0]
                    irrelevant_idx = dup_idx[1:]
                    dup_idx_del.append(irrelevant_idx)
                # Delete duplicate values from DIV data holders
                if len(dup_idx_del) > 0:
                    dup_idx_del = np.concatenate(dup_idx_del).ravel()
                    div_event = np.delete(div_event, dup_idx_del, axis=0)
                    div_dates = np.delete(div_dates, dup_idx_del, axis=0)
                    div_payday = np.delete(div_payday, dup_idx_del, axis=0)

                # Get correct indices
                div_dates_indices = np.where(np.in1d(dates_all, div_dates))[0]
                dates_div_indices = np.where(np.in1d(div_dates, dates_all))[0]
                # Update values
                div_event = div_event[dates_div_indices][::-1]
                div_payday = div_payday[dates_div_indices][::-1]
                # Fill values to matrices
                df_div_event[stock_idx, div_dates_indices] = div_event
                df_div_payday[stock_idx, div_dates_indices] = div_payday

            # Save DIV (event) to data dictionary.
            df_div_event = hl.ts_delay(df_div_event, 1)
            df_div_event = hl.zero_to_nan(hl.nan_to_zero(df_div_event))
            self.data['div.' + d_field.lower().split(' ')[0] + '.event'] = df_div_event

            # Save DIV (payday) to data dictionary.
            df_div_payday = hl.ts_delay(df_div_payday, 1)
            df_div_payday = hl.zero_to_nan(hl.nan_to_zero(df_div_payday))
            self.data['div.' + d_field.lower().split(' ')[0] + '.payday'] = df_div_payday
        # -----------------------------------------------------------------------------

        """
        # DATA ADJUSTMENT 
        # ----------------------------------------------------
        # Adjusting prices and volume by split factor and dividends (regular and special cash).       
        """
        # Define stock adjustment correction matrix.
        df_price_adjustment = np.ones_like(self.data['close'])
        df_volume_adjustment = np.ones_like(self.data['close'])
        # Go through each stock and fill stock adjustment matrix.
        for stock_idx, stock in enumerate(self.stocks_arr):
            # Skip if stock is irrelevant.
            if stock_idx in stocks_to_delete_idx:
                continue
            # Load div data.
            df_div = pd.read_csv(self.path_raw_div + stock + '.csv', header=0)
            df_div = df_div.sort_values(by=['Ex-Date'], ascending=False)
            div_events = df_div['Dividend Type'].values
            # Load price data.
            df_stock = pd.read_csv(self.path_raw_1D_data + 'close' + '/' + stock + '.csv', header=0)
            df_stock = df_stock.sort_values(by=['dt'])
            stock_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), df_stock['dt'].values)))

            # STOCK SPLIT
            # Get stock split information.
            split_values = df_div['Dividend Amount'].values[div_events == 'Stock Split']
            split_dates = df_div['Ex-Date'].values[div_events == 'Stock Split']
            split_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), split_dates)))
            split_record = df_div['Record Date'][div_events == 'Stock Split']
            # Define factor array, to adjust OHLCV data.
            split_factor_array = np.ones((dates_all.shape[0],), dtype=np.float64)
            split_factor = 1.0
            if len(split_values) != 0:
                # ---------------------------------------------------------------------
                # MISSING RECORD DATE
                # We check record date for stock split, if it`s missing
                # in stock data period we drop that stock. It`s not the best solution,
                # but it works and with low cost prevents from using totally wrong data.
                # ----------------------------------------------------------------------
                split_record_indices = np.argwhere(pd.isnull(split_record).values).ravel()
                split_record = split_dates[split_record_indices]
                split_record_in_dates = np.count_nonzero(np.in1d(stock_dates, split_record))
                if split_record_in_dates > 0:
                    stocks_to_delete_idx.append(stock_idx)
                    continue
                # Iterate through factors and fill the factor array.
                for split_idx, split_date in enumerate(split_dates):
                    split_factor = split_factor / split_values[split_idx]
                    split_factor_array[dates_all < split_date] = split_factor

            # REGULAR CASH
            # Get regular dividends information.
            div_reg_values = df_div['Dividend Amount'].values[div_events == 'Regular Cash']
            div_reg_dates = df_div['Ex-Date'].values[div_events == 'Regular Cash']
            div_reg_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), div_reg_dates)))
            # Define factor array, to adjust OHLC data.
            div_reg_factor_array = np.ones((dates_all.shape[0],), dtype=np.float64)
            div_reg_factor = 1.0
            if len(div_reg_values) != 0:
                # ------------------------------------------------------------
                # MISSING RECORD DATE
                # Brief analysis did not show any strong correlation between
                # record date existence and market reaction on dividend ex-date.
                # For the interview purposes we`ll miss that problem and we`ll
                # keep it for a future.
                # -------------------------------------------------------------
                # Iterate through factors and fill the factor array.
                for div_reg_idx, div_reg_date in enumerate(div_reg_dates):
                    # Get price on the day before ex-date
                    div_reg_date_idx = np.where(dates_all == div_reg_date)[0]
                    if len(div_reg_date_idx) != 0:
                        price_before = self.data['close'][stock_idx, div_reg_date_idx - 1]
                        # Adjust factor.
                        div_reg_factor *= (price_before - div_reg_values[div_reg_idx]) / price_before
                        div_reg_factor_array[dates_all < div_reg_date] = div_reg_factor

            # SPECIAL CASH
            # Get special cash information.
            div_spc_values = df_div['Dividend Amount'].values[div_events == 'Special Cash']
            div_spc_dates = df_div['Ex-Date'].values[div_events == 'Special Cash']
            div_spc_dates = np.array(list(map(lambda x: np.int(x.replace('-', '')), div_spc_dates)))
            # Define factor array, to adjust OHLCV data.
            div_spc_factor_array = np.ones((dates_all.shape[0],), dtype=np.float64)
            div_spc_factor = 1.0
            if len(div_spc_values) != 0:
                # ------------------------------------------------------------
                # MISSING RECORD DATE
                # Brief analysis did not show any strong correlation between
                # record date existence and market reaction on dividend ex-date.
                # For the interview purposes we`ll miss that problem and we`ll
                # keep it for a future.
                # -------------------------------------------------------------
                # Iterate through factors and fill the factor array.
                for div_spc_idx, div_spc_date in enumerate(div_spc_dates):
                    # Get price for the day before ex-date
                    div_spc_date_idx = np.where(dates_all == div_spc_date)[0]
                    if len(div_spc_date_idx) != 0:
                        price_before = self.data['close'][stock_idx, div_spc_date_idx - 1]
                        # Adjust factor.
                        div_spc_factor *= (price_before - div_spc_values[div_spc_idx]) / price_before
                        div_spc_factor_array[dates_all < div_spc_date] = div_spc_factor

            # Data adjustment arrays.
            volume_adj_factor_array = split_factor_array
            price_adj_factor_array = (split_factor_array * div_reg_factor_array * div_spc_factor_array)
            # Write adj. arrays to matrix.
            df_volume_adjustment[stock_idx, :] = volume_adj_factor_array
            df_price_adjustment[stock_idx, :] = price_adj_factor_array

        # Go through each field.
        # (adjust and save OHLCV data)
        for field in self.ohlcv_fields:
            df_ohlcv = np.copy(self.data[field])
            # Adjust data.
            if field == 'volume':
                df_ohlcv = df_ohlcv / df_volume_adjustment
            else:
                df_ohlcv = df_ohlcv * df_price_adjustment
            # Shift by (t+1) to avoid forward bias.
            df_ohlcv = hl.ts_delay(df_ohlcv, 1)
            df_ohlcv = hl.nan_to_zero(df_ohlcv)
            df_ohlcv = hl.zero_to_nan(df_ohlcv)
            # Save OHLCV to data dictionary.
            self.data[field] = df_ohlcv
        # ---------------------------------------------------

        # SAVE ADJUSTMENT MAT.
        df_volume_adjustment = hl.ts_delay(df_volume_adjustment, 1)
        df_volume_adjustment = hl.nan_to_zero(df_volume_adjustment)
        df_volume_adjustment = hl.zero_to_nan(df_volume_adjustment)
        self.data['adj.volume'] = df_volume_adjustment
        df_price_adjustment = hl.ts_delay(df_price_adjustment, 1)
        df_price_adjustment = hl.nan_to_zero(df_price_adjustment)
        df_price_adjustment = hl.zero_to_nan(df_price_adjustment)
        self.data['adj.price'] = df_price_adjustment

        # CLEAN
        # Get irrelevant stocks indices
        stocks_to_delete_idx = list(set(stocks_to_delete_idx))
        # Clean stocks list and delete irrelevant stocks data.
        self.stocks_arr = np.delete(self.stocks_arr, stocks_to_delete_idx, axis=0)
        # Go through each data holder and delete given idx.
        for field in self.data:
            self.data[field] = np.delete(self.data[field], stocks_to_delete_idx, axis=0)

        # SECTORS
        # Matrices containing information about sectors - industries - industryGroups.
        df_sectors = pd.read_csv(self.path_sector_codes, header=0, index_col=False)
        df_sectors = df_sectors.replace(np.nan, 'nan')
        df_sectors = df_sectors.sort_values(by=['TICKER'])
        df_sectors = df_sectors.reset_index(drop=True)
        # Select first sector out of two given (in any)
        for column in df_sectors.columns[:]:
            df_sectors[column] = df_sectors[column].apply(lambda x: x.split(',')[0])
            df_sectors[column] = df_sectors[column].apply(lambda x: x.replace('/', '.'))
        # Delete dropped stocks information.
        df_sectors_hold = np.in1d(df_sectors['TICKER'].values, self.stocks_arr)
        # Sector variables.
        industry_group = df_sectors['INDUSTRY_GROUP'].values[df_sectors_hold]
        industry_sector = df_sectors['INDUSTRY_SECTOR'].values[df_sectors_hold]
        industry_subgroup = df_sectors['INDUSTRY_SUBGROUP'].values[df_sectors_hold]

        # END
        # Fill data dict.
        self.data['dates'] = dates_all
        self.data['stocks'] = self.stocks_arr
        self.data['numdates'] = dates_all.shape[0]
        self.data['numstocks'] = self.stocks_arr.shape[0]
        self.data['industry.group'] = industry_group
        self.data['industry.sector'] = industry_sector
        self.data['industry.subgroup'] = industry_subgroup

        # Save data dict to disk.
        joblib.dump(self.data, self.path_data)
        print('  stocks: {} (end)'.format(len(self.stocks_arr)))
        print('data completed.')

    # --------------------------------------------------

    # Fn: (2)
    # Resample raw data to 1d format
    def resample_data(self, stocks, q, p_id):
        dates = set()
        for idx, stock in enumerate(stocks):
            # Declare columns to resample and method.
            ohlcv_dict = {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'}
            df = pd.read_csv(self.path_raw_15M_data + stock + '.csv', header=0, index_col=0)
            df.index = pd.to_datetime(df.index)
            # Resample every stock.
            df = df.resample('1D').apply(ohlcv_dict).dropna()
            dates = dates.union(df.index.values)
            # Save every field in stock data and save to separate folders.
            for field in self.ohlcv_fields:
                df[field].to_csv(path=self.path_raw_1D_data + field + '/' + stock + '.csv', header=True)
        dates = np.array(list(dates))
        q.put((p_id, dates))


if __name__ == '__main__':
    t = time.time()
    Data().prepare_data()
    print(time.time() - t)