# from __future__ import print_function
# import numpy as np
# import pandas as pd
# import configparser
# import warnings
# import helper as hl
# import datetime as dt
# import logging
# import os
# from sklearn.externals import joblib
# warnings.filterwarnings('ignore')
#
#
# class Simulation():
#
#     """
#     Run simulation
#     : input: predicted positions matrix
#     : return: sim object
#     : store: -
#
#     Brief description:
#         - PnL modeling on day to day basis.
#
#     Prepare data:
#         - Balancing long-short positions.
#         - Slowing down trading applying basic smoothing.
#         - Industry neutralization.
#         - Selecting N best stocks in both directions.
#
#     Details:
#         - Commissions: Interactive Brokers.
#         - Margin: Interactive Brokers.
#         - Price slippage: None
#
#     Assumptions:
#         - No borrowing costs.
#         - No borrowed cash for long positions.
#         - Immediate trade settlement.
#
#     """
#
#     def __init__(self, positions):
#
#         # Path to options file.
#         path_options = 'options.ini'
#
#         # Load config file.
#         config = configparser.ConfigParser()                                            # define config object.
#         config.optionxform = str                                                        # hold keys register
#         config.read(path_options)                                                       # read config file.
#
#         # Path variables.
#         self.path_data = config['PATH'].get('DataObj')                                  # address: data object
#         self.path_spx_price = config['PATH'].get('SPXDataPrice')                        # address: spx price
#         self.path_log = config['PATH'].get('LogObj')                                    # address: log object
#
#         # Load data
#         self.data = joblib.load(self.path_data)
#
#         # Define logger.
#         self.logger = self.get_logger()
#
#         # Sim params.
#         self.initial_capital = config['SIMULATION'].getfloat('InitialCapital')          # initial capital
#         self.n_best = config['SIMULATION'].getint('SelectNBest')                        # select n best stocks
#         self.turnover_limit = config['SIMULATION'].getfloat('TurnoverLimit')            # turnover limit
#         self.trading_volume_down = config['SIMULATION'].getfloat('TradingVolDown')      # volume down
#         self.smooth_positions = config['SIMULATION'].getint('PositionsSmooth')          # smoothing positions
#         self.first_index = self.data['first.index']                                     # sim start index
#
#         # Exchange commissions
#         self.commissions = dict()
#         self.commissions['size'] = 0.005                                                # in USD per share
#         self.commissions['min'] = 1.00                                                  # in USD per trade
#         self.commissions['max'] = 0.01                                                  # in % of trade volume.
#
#         # Load positions.
#         self.positions = positions[:, :-1]                                              # load and drop the last day
#
#     # Fn: (1)
#     # Run simulation based on positions matrix.
#     def run_simulation(self):
#
#         self.logger.info('')
#         self.logger.info('Simulation started:')
#         self.logger.info('Initial funds: ${:,.0f}'.format(self.initial_capital))
#         self.logger.info('')
#         self.logger.info('Description:')
#         self.logger.info('   PnL: pnl')
#         self.logger.info('   C:   cash')
#         self.logger.info('   L:   longs')
#         self.logger.info('   S:   shorts')
#         self.logger.info('   E:   equity')
#         self.logger.info('   AF:  available funds')
#         self.logger.info('   EL:  excess liquidity')
#         self.logger.info('   SMA: special memorandum account')
#         self.logger.info('   BP:  buying power')
#         self.logger.info('   L:   leverage')
#         self.logger.info('')
#
#         # Load dates arr.
#         dates = self.data['dates'][self.first_index:-1]
#
#         # Load market data.
#         spx = pd.read_csv(self.path_spx_price, header=0)['Close'].values.reshape(1, -1)
#         spx = hl.ts_delay(spx, 1)
#         spx = spx[:, self.first_index:-1]
#         spx_ret = ((spx - hl.ts_delay(spx, 1)) / hl.ts_delay(spx, 1)).ravel()
#         spx_ret[0] = spx_ret[1]
#
#         # Industry data. arr.
#         industry = self.data['industry.sector']
#
#         # Load open price mat.
#         open_price = self.data['open']
#         open_price = hl.ts_delay(open_price, -1)
#         open_price = open_price[:, self.first_index:-1]
#
#         # Load close price mat.
#         close_price = self.data['close']
#         close_price = hl.ts_delay(close_price, -1)
#         close_price = close_price[:, self.first_index:-1]
#
#         # Positions mat. pre-processing.
#         positions_mat = self.positions
#         positions_mat = hl.fn_booksize(positions_mat, booksize=1e6)                 # balance l/s and apply booksize
#         positions_mat = hl.ts_mean(hl.zero_to_nan(positions_mat),                   # smoothing positions
#                                    window=self.smooth_positions, hold_first=True)   # ......
#         positions_mat = hl.nan_to_zero(positions_mat)                               # ......
#         positions_mat = hl.fn_industry_neutralization(positions_mat, industry)      # industry neutralization
#         positions_mat = hl.fn_select_n_best(positions_mat, self.n_best)             # select N best stocks
#         positions_mat = hl.fn_positions_go2pct(positions_mat)                       # convert positions to %
#
#         # Check mat. sizes.
#         try:
#             assert (dates.shape[0] == positions_mat.shape[1])
#             assert (dates.shape[0] == spx_ret.shape[0])
#             assert (open_price.shape == positions_mat.shape)
#             assert (close_price.shape == positions_mat.shape)
#         except AssertionError:
#             print('Data shapes doesn`t match.')
#             raise
#
#         # Create portfolio object.
#         # -------------------------------------------------------
#         # Portfolio object store all necessary information about
#         # all transactions. Object updated on day to day basis.
#         # -------------------------------------------------------
#         portfolio = AcPortfolio(self.data, self.positions, dates, self.initial_capital, spx_ret)
#
#         # Go through each trading day.
#         for i, date in enumerate(dates):
#
#             # Price data.
#             open_today = self.get_open_price_today(open_price, close_price, i)
#             close_today = self.get_close_price_today(close_price, i)
#             valids = self.get_valids_mask(open_price, i)
#
#             # Update portfolio (start of day).
#             # ---------------------------------------------------------
#             # 1. Use open price data to update account portfolio.
#             # 2. Update portfolio on dividend payments (+/-)
#             # 3. Account validaty control:
#             #        - check minimum equity requirement
#             #          (sim terminated if requirement check is failed)
#             # ---------------------------------------------------------
#             portfolio = self.start_of_day_portfolio_update(portfolio, open_today, date, i)
#
#             # Trade order.
#             # ---------------------------------------------------------
#             # Create trade order based on current account equity and
#             # returns two objects: target and current trading list.
#             # ---------------------------------------------------------
#             target_list, trade_list = self.build_trade_order(portfolio, positions_mat, valids, open_today, i)
#
#             # Execute sell order (vector implementation)
#             # ---------------------------------------------------------
#             # 1. Balance portfolio according to defined target order:
#             #       - close existing longs and open new short positions.
#             # 2. Account validaty control:
#             #       - check minimum equity requirement
#             #         (sim terminated if requirement check is failed)
#             # 3. Trade validaty control:
#             #       - check if trade order acceptable or not via
#             #         (hard cash, available funds and leverage) check.
#             # ---------------------------------------------------------
#             portfolio = self.execute_sell_order(portfolio, target_list, trade_list, open_today, i)
#
#             # Execute buy order (vector implementation)
#             # ---------------------------------------------------------
#             # 1. Balance portfolio according to defined target order:
#             #       - cover existing shorts and open new long positions.
#             # 2. Account validaty control:
#             #       - check minimum equity requirement
#             #         (sim terminated if requirement check is failed)
#             # 3. Trade validaty control:
#             #       - check if trade order acceptable or not via
#             #         (hard cash, available funds and leverage) check.
#             # ---------------------------------------------------------
#             portfolio = self.execute_buy_order(portfolio, target_list, trade_list, open_today, i)
#
#             # Update portfolio (end of day).
#             # ---------------------------------------------------------
#             # 1. Use close price data to update account portfolio.
#             # 2. Account validaty control:
#             #       - check minimum equity requirement
#             #         (sim terminated if requirement check is failed)
#             #       - check if account valid via
#             #         (SMA and excess liquidity) check.
#             # 3. If account not valid positions liquidation order executed.
#             # ---------------------------------------------------------
#             portfolio = self.end_of_day_portfolio_update(portfolio, close_today, i)
#             # ---------------------------------------------------------
#
#         # Return simulation results.
#         simres = portfolio.get_results()
#         self.logger.info('')
#         self.logger.info('Final portfolio value: ${:,.0f}'.format(portfolio.balance['equity'][-1]))
#         self.logger.info('  simulation log saved')
#         self.logger.info('  simulation completed')
#         return simres
#         # -----------------------------------
#
#     # Fn: (1)
#     # Define logger.
#     def get_logger(self):
#         # Define logger params.
#         logger_name = dt.datetime.now().strftime('%Y-%m-%d %H-%M-%S')
#         logger = logging.getLogger(logger_name)
#         # Logger level.
#         logger.setLevel(logging.INFO)
#         logger_path = self.path_log
#         logger_file = '%ssim-%s.txt' % (logger_path, logger_name)
#         # Create folder to store sim log.
#         if not os.path.exists(logger_path):
#             os.makedirs(logger_path)
#         formatter = logging.Formatter('%(message)s')
#         file_handler = logging.FileHandler(logger_file)
#         console_handler = logging.StreamHandler()
#         file_handler.setFormatter(formatter)
#         console_handler.setFormatter(formatter)
#         logger.addHandler(file_handler)
#         logger.addHandler(console_handler)
#         return logger
#
#     # Fn: (2)
#     # Log portfolio statement.
#     def logging(self, portfolio, day):
#         # Derive required balance positions.
#         pnl = portfolio.balance['pnl'][day]
#         cash = portfolio.balance['cash'][day]
#         longs = np.sum(portfolio.balance['value'][:, day][portfolio.balance['value'][:, day] > 0])
#         shorts = np.sum(portfolio.balance['value'][:, day][portfolio.balance['value'][:, day] < 0])
#         equity = portfolio.balance['equity'][day]
#         available = portfolio.funds['available'][day]
#         excess = portfolio.funds['excess'][day]
#         sma = portfolio.funds['sma'][day]
#         bp = portfolio.funds['bp'][day]
#         leverage = portfolio.funds['leverage'][day]
#         # Set up logging object.
#         s = '' \
#             'PnL: {:,.0f}. ' \
#             'C: {:,.0f}. ' \
#             'L: {:,.0f}. ' \
#             'S: {:,.0f}. ' \
#             'E: {:,.0f}. ' \
#             'AF: {:,.0f}. ' \
#             'EL: {:,.0f}. ' \
#             'SMA: {:,.0f}. ' \
#             'BP: {:,.0f}. ' \
#             'L: {:.2f}.'. \
#             format(pnl, cash, longs, shorts, equity, available, excess, sma, bp, leverage)
#         # Log.
#         self.logger.info(s)
#
#     # Fn: (3)
#     # Get open price.
#     # TODAY.
#     @staticmethod
#     def get_open_price_today(price, close, day):
#         # Daily data.
#         open_today = np.copy(price[:, day])
#         # It`s hard to predict if any stock will exist next day or not in
#         # basic simulation. To prevent situations when we sell the stock
#         # @0.0 we will use a little magic and will sell delisted
#         # stock @ previous day close price. It`s not the best decision but
#         # anyway it should work and should not strongly affect on performance.
#         close_price_yesterday = np.copy(close[:, np.maximum(day - 1, 0)])                   # last close price
#         # Following that idea we unite prices.                                              # ....
#         open_today[np.isnan(open_today)] = close_price_yesterday[np.isnan(open_today)]      # unite prices
#         open_today = hl.nan_to_zero(open_today)  # ....
#         # Return results
#         return open_today
#
#     # Fn: (4)
#     # Get close price.
#     # TODAY.
#     @staticmethod
#     def get_close_price_today(price, day):
#         return hl.nan_to_zero(np.copy(price[:, day]))
#
#     # Fn: (5)
#     # Get valids mask.
#     @staticmethod
#     def get_valids_mask(price, day):
#         open_today = np.copy(price[:, day])
#         valids = ~np.isnan(open_today)
#         return valids
#
#     # Fn: (6)
#     # Build trade order with positions list.
#     def build_trade_order(self, portfolio, positions_predicted, valids, price, day):
#         # Check target positions for valids -->
#         # We`ll not open positions in stocks if there is no price
#         # and additionally it will force us to close positions in
#         # unavailable stocks (@yesterday closing price)
#         target_list = hl.fn_positions_go2pct(positions_predicted[:, day] * valids)
#         # Positions to open needed to balance portfolio according the model prediction.
#         target_list = ((target_list * portfolio.balance['equity'][day]) / price)
#         target_list = hl.nan_to_zero(hl.zero_to_nan(target_list))
#         target_list = target_list.astype(np.int32)
#         # Positions to trade today.
#         trade_list = (target_list - portfolio.balance['positions'][:, day])
#         trade_list = np.multiply(trade_list, (self.turnover_limit +
#                                               (1 - self.turnover_limit) * (not bool(day))))
#         trade_list = trade_list.astype(np.int32)
#         return target_list, trade_list
#
#     # Fn: (7)
#     # Build trade order with positions list.
#     # In case of positions liquidation.
#     @staticmethod
#     def build_liquidation_order(portfolio, liquidation_size, price, day):
#         # Portfolio actual positions.
#         actual_list = portfolio.balance['positions'][:, day]
#         # Balance liquidation size across positions
#         # according to current concentration.
#         trade_list = hl.fn_positions_go2pct(actual_list)
#         # Positions to trade to balance margin requirements.
#         trade_list = (trade_list * (liquidation_size * 0.5) * (1.0)) / price
#         trade_list = hl.nan_to_zero(hl.zero_to_nan(trade_list))
#         # Round up trade order units to be sure that margin
#         # requirement is satisfied.
#         long_list = np.ceil(trade_list * (trade_list > 0))
#         short_list = np.ceil(np.abs(trade_list * (trade_list < 0))) * (-1.0)
#         # Final trade list.
#         trade_list = (long_list + short_list).astype(np.int32)
#         # Positions required to balance margin.
#         target_list = actual_list - trade_list
#         return target_list, trade_list
#
#     # Fn: (8)
#     # Start of day portfolio update.
#     def start_of_day_portfolio_update(self, portfolio, price, date, day):
#
#         # Log start of day.
#         self.logger.info('Trading date: {}. (day: {})'.
#                          format(dt.datetime.strptime(str(int(date)), '%Y%m%d'), day))
#
#         # First trading day already initialized
#         # with appropriate values in Portfolio class.
#         if day > 0:
#
#             # Cash flows (balance).
#             dividends = portfolio.get_dividends(day)
#             cash = portfolio.balance['cash'][day-1] + dividends
#             positions = portfolio.balance['positions'][:, day-1]
#             value = AcStatements.get_value(positions, price)
#             longs = AcStatements.get_longs(value)
#             shorts = AcStatements.get_shorts(value)
#             hard_cash = AcStatements.get_hard_cash(cash, shorts)
#             equity = AcStatements.get_equity(cash, longs, shorts)
#             pnl = AcStatements.get_pnl(portfolio, equity, (day - 1))
#
#             # Cash flows (margin).
#             initial = AcStatements.get_initial_margin(portfolio, longs, shorts)
#             maintenance = AcStatements.get_maintenance_margin(portfolio, positions, price)
#             regT_total = AcStatements.get_regT_total(portfolio, longs, shorts)
#             regT_current = 0
#
#             # Cash flows (funds)
#             available = AcStatements.get_available_funds(equity, initial)
#             excess = AcStatements.get_excess_liquidity(equity, maintenance)
#             sma = AcStatements.get_sma(portfolio, equity, regT_total, regT_current, (day - 1), dividends, comm=0)
#             bp = AcStatements.get_buying_power(portfolio, sma, excess)
#             leverage = AcStatements.get_leverage(equity, longs, shorts)
#
#             # Portfolio (balance).
#             portfolio.balance['cash'][day] = cash                                       # update: cash
#             portfolio.balance['hard_cash'][day] = hard_cash                             # update: hard cash
#             portfolio.balance['dividends'][day] += dividends                            # update: dividends
#             portfolio.balance['positions'][:, day] = positions                          # update: positions
#             portfolio.balance['value'][:, day] = value                                  # update: value
#             portfolio.balance['equity'][day] = equity                                   # update: equity
#             portfolio.balance['pnl'][day] += pnl                                        # update: pnl
#             # Portfolio (margin).
#             portfolio.margin['initial'][day] = initial                                  # update: initial margin
#             portfolio.margin['maintenance'][day] = maintenance                          # update: min margin
#             portfolio.margin['regT_total'][day] = regT_total                            # update: regT total req.
#             portfolio.margin['regT_current'][day] = regT_current                        # update: regT today req.
#             # Portfolio (funds).
#             portfolio.funds['available'][day] = available                               # update: available funds
#             portfolio.funds['excess'][day] = excess                                     # update: excess liq.
#             portfolio.funds['sma'][day] = sma                                           # update: sma
#             portfolio.funds['bp'][day] = bp                                             # update: buying power
#             portfolio.funds['leverage'][day] = leverage                                 # update: leverage
#             # Portfolio (details).
#             portfolio.details['commission'][day] += 0                                   # update: commissions
#             portfolio.details['slippage'][day] += 0                                     # update: price slippage
#             portfolio.details['turnover'][day] += 0                                     # update: turnover
#
#         # Update portfolio attributes
#         portfolio.copy_equity(day)
#         portfolio.update_safe_heaven(day)
#         self.logging(portfolio, day)
#         return portfolio
#
#     # Fn: (9)
#     # End of day portfolio update.
#     def end_of_day_portfolio_update(self, portfolio, price, day):
#
#         # Check initial equity requirement.
#         AcValidaty.check_initial_equity_requirement(self.logger, portfolio, day)
#
#         # Cash flows (balance).
#         cash = portfolio.balance['cash'][day]
#         positions = portfolio.balance['positions'][:, day]
#         value = AcStatements.get_value(positions, price)
#         longs = AcStatements.get_longs(value)
#         shorts = AcStatements.get_shorts(value)
#         hard_cash = AcStatements.get_hard_cash(cash, shorts)
#         equity = AcStatements.get_equity(cash, longs, shorts)
#         pnl = AcStatements.get_pnl(portfolio, equity, day)
#
#         # Cash flows (margin).
#         initial = AcStatements.get_initial_margin(portfolio, longs, shorts)
#         maintenance = AcStatements.get_maintenance_margin(portfolio, positions, price)
#         regT_total = AcStatements.get_regT_total(portfolio, longs, shorts)
#         regT_current = 0
#
#         # Cash flows (funds)
#         available = AcStatements.get_available_funds(equity, initial)
#         excess = AcStatements.get_excess_liquidity(equity, maintenance)
#         sma = AcStatements.get_sma(portfolio, equity, regT_total, regT_current, day, divs=0, comm=0)
#         bp = AcStatements.get_buying_power(portfolio, sma, excess)
#         leverage = AcStatements.get_leverage(equity, longs, shorts)
#
#         # Portfolio (balance).
#         portfolio.balance['cash'][day] = cash                                           # update: cash
#         portfolio.balance['hard_cash'][day] = hard_cash                                 # update: hard cash
#         portfolio.balance['dividends'][day] += 0                                        # update: dividends
#         portfolio.balance['positions'][:, day] = positions                              # update: positions
#         portfolio.balance['value'][:, day] = value                                      # update: value
#         portfolio.balance['equity'][day] = equity                                       # update: equity
#         portfolio.balance['pnl'][day] += pnl                                            # update: pnl
#         # Portfolio (margin).
#         portfolio.margin['initial'][day] = initial                                      # update: initial margin
#         portfolio.margin['maintenance'][day] = maintenance                              # update: min margin
#         portfolio.margin['regT_total'][day] = regT_total                                # update: regT total req.
#         portfolio.margin['regT_current'][day] = regT_current                            # update: regT today req.
#         # Portfolio (funds).
#         portfolio.funds['available'][day] = available                                   # update: available funds
#         portfolio.funds['excess'][day] = excess                                         # update: excess liq.
#         portfolio.funds['sma'][day] = sma                                               # update: sma
#         portfolio.funds['bp'][day] = bp                                                 # update: buying power
#         portfolio.funds['leverage'][day] = leverage                                     # update: leverage
#         # Portfolio (details).
#         portfolio.details['commission'][day] += 0                                       # update: commissions
#         portfolio.details['slippage'][day] += 0                                         # update: price slippage
#         portfolio.details['turnover'][day] += 0                                         # update: turnover
#
#         # Check SMA and excess liquidity requirements.
#         account_validaty = AcValidaty.check_account_validaty(self.logger, excess, sma)
#         if not account_validaty:
#             self.logger.debug('Started positions liquidation.')
#             self.run_liquidation_transaction(portfolio, price, day)
#
#         # Log.
#         self.logging(portfolio, day)
#         self.logger.info('-' * 120)
#         return portfolio
#
#     # Fn: (10)
#     # Sell order.
#     def execute_sell_order(self, portfolio, target_list, trade_list, price, day):
#         # Create trades object to store all
#         # necessary trades for further functions.
#         trades = dict()
#         # Build trade orders based on desired trade direction.
#         trade_list = trade_list * (trade_list < 0)
#         actual_list = portfolio.balance['positions'][:, day]
#         # In case of order rejection due to margin requirements
#         # execution function will reduce open order size.
#         mO = (target_list < 0) * (trade_list < 0)
#         open_list = target_list * mO - actual_list * mO * (actual_list < 0)
#         # Fill trades object.
#         trades['actual_list'] = actual_list
#         trades['trade_list'] = trade_list
#         trades['open_list'] = open_list
#         trades['target_list'] = target_list
#         # Execute trade based on trade orders.
#         portfolio = self.run_trade_transaction(portfolio, trades, price, day)
#         return portfolio
#
#     # Fn: (11)
#     # Buy order.
#     def execute_buy_order(self, portfolio, target_list, trade_list, price, day):
#         # Create trades object to store all
#         # necessary trades for further functions.
#         trades = dict()
#         # Build trade orders based on desired trade direction.
#         trade_list = trade_list * (trade_list > 0)
#         actual_list = portfolio.balance['positions'][:, day]
#         # In case of order rejection due to margin requirements
#         # execution function will reduce open order size.
#         mO = (target_list > 0) * (trade_list > 0)
#         open_list = target_list * mO - actual_list * mO * (actual_list > 0)
#         # Fill trades object.
#         trades['actual_list'] = actual_list
#         trades['trade_list'] = trade_list
#         trades['open_list'] = open_list
#         trades['target_list'] = target_list
#         # Execute trade based on trade orders.
#         portfolio = self.run_trade_transaction(portfolio, trades, price, day)
#         return portfolio
#
#     # Fn: (12)
#     # Execute trade transaction.
#     def run_trade_transaction(self, portfolio, trades, price, day):
#         # Check initial equity requirement.
#         AcValidaty.check_initial_equity_requirement(self.logger, portfolio, day)
#         # Run trade transaction.
#         portfolio = self.execute_trade(portfolio, trades, price, day, tvd=0)
#         # Log trade transaction.
#         self.logging(portfolio, day)
#         return portfolio
#
#     # Fn: (13)
#     # Execute trade transaction.
#     # In case of positions liquidation.
#     def run_liquidation_transaction(self, portfolio, price, day):
#         # Define liquidation size to match margin requirements.
#         liquidation_size = (portfolio.funds['excess'][day] / portfolio.maintenance_margin) * (-1.0)
#         # Build liquidation trade order.
#         target_list, trade_list = self.build_liquidation_order(portfolio, liquidation_size, price, day)
#         # Call sell - buy execution functions to trade order.
#         self.execute_sell_order(portfolio, target_list, trade_list, price, day)
#         self.execute_buy_order(portfolio, target_list, trade_list, price, day)
#         # Update portfolio statements.
#         self.end_of_day_portfolio_update(portfolio, price, day)
#
#     # Fn: (14)
#     # Execute trade.
#     def execute_trade(self, portfolio, trades, price, day, tvd):
#
#         # Decrease trade order size.
#         decrease = np.multiply(trades['open_list'], tvd).astype(np.int32)
#         trade_order = trades['trade_list'] - decrease
#
#         # Update trades object.
#         trades['target_list'] -= decrease
#         trades['trade_list'] -= decrease
#
#         # Cost of trade.
#         slippage = AcStatements.get_price_slippage(portfolio, price, day)
#         price = price + slippage
#         trade_order_size = trade_order * price
#         commission = AcStatements.get_commission(self.commissions,
#                                                  trade_order, trade_order_size)
#         cost_of_trade = np.sum(trade_order_size)
#
#         # Cash flows.
#         cash = portfolio.balance['cash'][day] - cost_of_trade - commission
#         positions = portfolio.balance['positions'][:, day] + trade_order
#         value = AcStatements.get_value(positions, price)
#         longs = AcStatements.get_longs(value)
#         shorts = AcStatements.get_shorts(value)
#         hard_cash = AcStatements.get_hard_cash(cash, shorts)
#         equity = AcStatements.get_equity(cash, longs, shorts)
#         pnl = AcStatements.get_pnl(portfolio, equity, day)
#
#         # Cash flows (margin).
#         initial = AcStatements.get_initial_margin(portfolio, longs, shorts)
#         maintenance = AcStatements.get_maintenance_margin(portfolio, positions, price)
#         regT_total = AcStatements.get_regT_total(portfolio, longs, shorts)
#         regT_current = AcStatements.get_regT_current(portfolio, trades, price)
#
#         # Cash flows (funds)
#         available = AcStatements.get_available_funds(equity, initial)
#         excess = AcStatements.get_excess_liquidity(equity, maintenance)
#         sma = AcStatements.get_sma(portfolio, equity, regT_total, regT_current,
#                                    day, divs=0, comm=commission)
#         bp = AcStatements.get_buying_power(portfolio, sma, excess)
#         leverage = AcStatements.get_leverage(equity, longs, shorts)
#
#         # Check order validaty.
#         order_validaty = AcValidaty.check_order_validaty(self.logger, portfolio,
#                                                          hard_cash, available, leverage)
#
#         if not order_validaty:
#             tvd += self.trading_volume_down
#             return self.execute_trade(portfolio, trades, price, day, tvd)
#         else:
#             # Portfolio (balance).
#             portfolio.balance['cash'][day] = cash                                       # update: cash
#             portfolio.balance['hard_cash'][day] = hard_cash                             # update: hard cash
#             portfolio.balance['dividends'][day] += 0                                    # update: dividends
#             portfolio.balance['positions'][:, day] = positions                          # update: positions
#             portfolio.balance['value'][:, day] = value                                  # update: value
#             portfolio.balance['equity'][day] = equity                                   # update: equity
#             portfolio.balance['pnl'][day] += pnl                                        # update: pnl
#             # Portfolio (margin).
#             portfolio.margin['initial'][day] = initial                                  # update: initial margin
#             portfolio.margin['maintenance'][day] = maintenance                          # update: min margin
#             portfolio.margin['regT_total'][day] = regT_total                            # update: regT total req.
#             portfolio.margin['regT_current'][day] = regT_current                        # update: regT today req.
#             # Portfolio (funds).
#             portfolio.funds['available'][day] = available                               # update: available funds
#             portfolio.funds['excess'][day] = excess                                     # update: excess liq.
#             portfolio.funds['sma'][day] = sma                                           # update: sma
#             portfolio.funds['bp'][day] = bp                                             # update: buying power
#             portfolio.funds['leverage'][day] = leverage                                 # update: leverage
#             # Portfolio (details).
#             portfolio.details['commission'][day] += commission                          # update: commission
#             portfolio.details['slippage'][day] += np.sum(slippage)                      # update: price slippage
#             portfolio.details['turnover'][day] += np.abs(cost_of_trade                  # update: turnover
#                                                          / portfolio.equity)            # .....
#             return portfolio
#
#
# class AcPortfolio():
#
#     """
#     Account portfolio.
#     : input: -
#     : return: portfolio.
#     : store: -
#
#     Brief description:
#         - Creates class used to model account statements and
#           trade activity via simulation class. Store all necessary data
#           and returns it in dict.
#
#     Details:
#         - Stores all dividends data.
#
#     """
#
#     def __init__(self, data_obj, predictions, dates, initial_capital, market):
#
#         # Margin account options.
#         self.min_capital = 2000
#         self.initial_margin = 0.5
#         self.maintenance_margin = 0.3
#         self.regT_margin = 0.5
#
#         # Initialize attributes
#         self.safe_heaven = 0
#         self.safe_heaven_period = 10
#         self.market = market
#         self.equity = 0
#
#         # Balance statement.
#         self.balance = dict()
#         self.balance['cash'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.balance['hard_cash'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.balance['dividends'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.balance['positions'] = np.zeros_like(predictions, dtype=np.float32)
#         self.balance['value'] = np.zeros_like(predictions, dtype=np.float32)
#         self.balance['equity'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.balance['pnl'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#
#         # Margin statement.
#         self.margin = dict()
#         self.margin['regT_total'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.margin['regT_current'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.margin['initial'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.margin['maintenance'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#
#         # Funds statement.
#         self.funds = dict()
#         self.funds['available'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.funds['excess'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.funds['sma'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.funds['bp'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.funds['leverage'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#
#         # Details statement.
#         self.details = dict()
#         self.details['commission'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.details['slippage'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#         self.details['turnover'] = np.zeros((predictions.shape[1],), dtype=np.float32)
#
#         # Initialize portfolio state at first day.
#         self.balance['cash'][0] = initial_capital
#         self.balance['equity'][0] = initial_capital
#         self.funds['available'][0] = initial_capital
#         self.funds['excess'][0] = initial_capital
#         self.funds['leverage'][0] = 1.0
#         self.funds['sma'][0] = initial_capital
#         self.funds['bp'][0] = initial_capital * 2
#
#         # Prepare dividends data obj.
#         self.dividends = dict()
#         self.prepare_dividends_data(data_obj, predictions, dates)
#
#     # Fn: (1)
#     # Prepare dividends data holder.
#     def prepare_dividends_data(self, data_obj, predictions, dates):
#
#         # Load regular cash dividends mat.
#         div_regular = data_obj['div.regular.event']
#         div_regular = hl.ts_delay(div_regular, -1)
#         div_regular = div_regular[:, data_obj['first.index']:-1]
#         div_regular = hl.nan_to_zero(div_regular)
#
#         # Load special cash dividends mat.
#         div_special = data_obj['div.special.event']
#         div_special = hl.ts_delay(div_special, -1)
#         div_special = div_special[:, data_obj['first.index']:-1]
#         div_special = hl.nan_to_zero(div_special)
#
#         # Load regular cash dividend payment dates.
#         div_regular_payday = data_obj['div.regular.payday']
#         div_regular_payday = hl.ts_delay(div_regular_payday, -1)
#         div_regular_payday = div_regular_payday[:, data_obj['first.index']:-1]
#         div_regular_payday = hl.nan_to_zero(div_regular_payday)
#
#         # Load special cash dividend payment dates.
#         div_special_payday = data_obj['div.special.payday']
#         div_special_payday = hl.ts_delay(div_special_payday, -1)
#         div_special_payday = div_special_payday[:, data_obj['first.index']:-1]
#         div_special_payday = hl.nan_to_zero(div_special_payday)
#
#         # Load data adjustment mat.
#         adj_price = data_obj['adj.price']
#         adj_price = hl.ts_delay(adj_price, -1)
#         adj_price = adj_price[:, data_obj['first.index']:-1]
#         adj_price = hl.nan_to_zero(adj_price)
#
#         # Write dividends data to dict.
#         self.dividends['dates'] = dates
#         self.dividends['div_regular'] = div_regular
#         self.dividends['div_special'] = div_special
#         self.dividends['div_regular_payday'] = div_regular_payday
#         self.dividends['div_special_payday'] = div_special_payday
#         self.dividends['adj_price'] = adj_price
#         self.dividends['regular_received'] = np.zeros_like(predictions, dtype=np.float32)
#         self.dividends['special_received'] = np.zeros_like(predictions, dtype=np.float32)
#
#     # Fn: (2)
#     # Get dividends.
#     def get_dividends(self, day):
#         # Dividends to receive and pay on today.
#         div_received = self.get_dividends_to_receive(day)
#         div_payed = self.get_dividends_to_pay(day)
#         # Total dividends.
#         div_total = div_received - div_payed
#         return div_total
#
#     # Fn: (3)
#     # Compute dividends to receive.
#     def get_dividends_to_receive(self, day):
#         # Positions in portfolio.
#         longs = self.balance['positions'][:, day - 1] * (self.balance['positions'][:, day - 1] > 0)
#         # Dividends today.
#         div_regular_today = self.dividends['div_regular'][:, day] * self.dividends['adj_price'][:, day]
#         div_special_today = self.dividends['div_special'][:, day] * self.dividends['adj_price'][:, day]
#         # Dividends to receive in future.
#         div_regular_future = np.multiply(div_regular_today, longs)
#         div_special_future = np.multiply(div_special_today, longs)
#         # Update future payments mat.
#         for stock in range(div_regular_future.shape[0]):
#             regular_index = np.where(self.dividends['dates'] == self.dividends['div_regular_payday'][stock, day])[0]
#             special_index = np.where(self.dividends['dates'] == self.dividends['div_special_payday'][stock, day])[0]
#             self.dividends['regular_received'][stock, regular_index] = div_regular_future[stock]
#             self.dividends['special_received'][stock, special_index] = div_special_future[stock]
#         # Dividends to receive on opened long positions
#         # (position: Ex-Date) (payment: payday)
#         div_received = np.sum(self.dividends['regular_received'][:, day] + self.dividends['special_received'][:, day])
#         return div_received
#
#     # Fn: (4)
#     # Compute dividends to pay.
#     def get_dividends_to_pay(self, day):
#         # Short positions in portfolio.
#         shorts = self.balance['positions'][:, day - 1] * (self.balance['positions'][:, day - 1] < 0)
#         # Dividends today (regular and special cash)
#         div_regular_today = self.dividends['div_regular'][:, day] * self.dividends['adj_price'][:, day]
#         div_special_today = self.dividends['div_special'][:, day] * self.dividends['adj_price'][:, day]
#         # Dividends to pay on opened short positions
#         # (position: Ex-Date) (payment: today)
#         div_payed = np.sum(np.abs((div_regular_today + div_special_today) * shorts))
#         return div_payed
#
#     # Fn: (5)
#     # Update equity mirror.
#     def copy_equity(self, day):
#         self.equity = np.copy(self.balance['equity'][day])
#
#     # Fn: (6)
#     # Get market std.
#     def get_market_std(self, day):
#         market_std = np.std(self.market[np.maximum(day - self.safe_heaven_period, 0):day])
#         return market_std
#
#     # Fn: (7)
#     # Update safe heaven value.
#     def update_safe_heaven(self, day):
#         if day > 0:
#             self.safe_heaven = (self.get_market_std(day) * self.balance['equity'][day]) * 1.0
#
#     # Fn: (8)
#     # Return results object.
#     def get_results(self):
#         # Portfolio.
#         portfolio = dict()
#         portfolio['positions'] = self.balance['positions']
#         portfolio['value'] = self.balance['value']
#         portfolio['cash'] = self.balance['cash']
#         portfolio['equity'] = self.balance['equity']
#         portfolio['pnl'] = self.balance['pnl']
#         portfolio['turnover'] = self.details['turnover']
#         # Trade cost.
#         trade_cost = dict()
#         trade_cost['exchange_commission'] = self.details['commission']
#         trade_cost['price_slippage'] = self.details['slippage']
#         # Return results.
#         simres = {'portfolio': portfolio, 'trade_cost': trade_cost}
#         return simres
#
#
# class AcValidaty:
#
#     """
#     Account validaty.
#     : input: sim class objects.
#     : return: boolean marks.
#     : store: -
#
#     Brief description:
#         - Helper class containing functions used to
#           check account and trade orders on validaty.
#
#     """
#
#     # Fn: (1)
#     # Check initial margin requirement.
#     @staticmethod
#     def check_initial_equity_requirement(logger, portfolio, day):
#         try:
#             assert portfolio.balance['equity'][day] >= portfolio.min_capital
#         except AssertionError:
#             logger.warning('Sim stopped. Equity is less then required 2000$')
#             raise
#
#     # Fn: (2)
#     # Check order validaty.
#     @staticmethod
#     def check_order_validaty(logger, portfolio, hard_cash, available, leverage):
#         if available < -1e3:
#             return True
#         try:
#             assert hard_cash >= 0
#             assert available >= portfolio.safe_heaven
#             assert leverage >= portfolio.maintenance_margin
#             return True
#         except AssertionError:
#             logger.debug('Order rejected. Reduce trade size to match requirements.')
#             return False
#
#     # Fn: (3)
#     # Check account validaty.
#     @staticmethod
#     def check_account_validaty(logger, excess, sma):
#         try:
#             assert excess >= 0
#             assert sma >= 0
#             return True
#         except AssertionError:
#             logger.debug('Margin requirement is not satisfied. Positions liquidation required.')
#             return False
#
#
# class AcStatements:
#
#     """
#     Account statements.
#     : input: sim class object.
#     : return: statements.
#     : store: -
#
#     Brief description:
#         - Helper class containing functions used to
#           compute required account statements.
#
#     """
#
#     # Fn: (1)
#     # Value.
#     @staticmethod
#     def get_value(positions, price):
#         value = positions * price
#         return value
#
#     # Fn: (2)
#     # Longs (in USD).
#     @staticmethod
#     def get_longs(value):
#         longs = np.sum(np.abs(np.multiply(value, (value > 0))))
#         return longs
#
#     # Fn: (3)
#     # Shorts (in USD).
#     @staticmethod
#     def get_shorts(value):
#         shorts = np.sum(np.abs(np.multiply(value, (value < 0))))
#         return shorts
#
#     # Fn: (4)
#     # Hard Cash.
#     @staticmethod
#     def get_hard_cash(cash, shorts):
#         hard_cash = cash - shorts
#         return hard_cash
#
#     # Fn: (5)
#     # Equity.
#     @staticmethod
#     def get_equity(cash, longs, shorts):
#         equity = cash + longs - shorts
#         return equity
#
#     # Fn: (6)
#     # PnL.
#     @staticmethod
#     def get_pnl(portfolio, equity, day):
#         pnl = equity - portfolio.balance['equity'][day]
#         return pnl
#
#     # Fn: (7)
#     # Initial Margin.
#     @staticmethod
#     def get_initial_margin(portfolio, longs, shorts):
#         initial = longs * portfolio.initial_margin + shorts * portfolio.initial_margin
#         return initial
#
#     # Fn: (8)
#     # Maintenance Margin.
#     @staticmethod
#     def get_maintenance_margin(portfolio, positions, price):
#         # Derive long-shorts.
#         longs = np.abs(np.multiply(positions, (positions > 0)))
#         shorts = np.abs(np.multiply(positions, (positions < 0)))
#         # Price masks.
#         tier_1 = (price > 16.66)
#         tier_2 = (price > 5.00) * (price < 16.66)
#         tier_3 = (price > 2.50) * (price < 4.99)
#         tier_4 = (price < 2.50)
#         # Define margin.
#         maintenance = longs + shorts
#         maintenance[tier_1] = maintenance[tier_1] * price[tier_1] \
#                               * portfolio.maintenance_margin
#         maintenance[tier_2] = maintenance[tier_2] * 5.00
#         maintenance[tier_3] = maintenance[tier_3] * price[tier_3]
#         maintenance[tier_4] = maintenance[tier_4] * 2.50
#         # Total maintenance margin.
#         maintenance = np.sum(maintenance)
#         return maintenance
#
#     # Fn: (9)
#     # RegT Margin.
#     @staticmethod
#     def get_regT_total(portfolio, longs, shorts):
#         regT_total = longs * portfolio.regT_margin + shorts * portfolio.regT_margin
#         return regT_total
#
#     # Fn: (10)
#     # RegT Margin (today).
#     @staticmethod
#     def get_regT_current(portfolio, trades, price):
#         # Trade lists.
#         actual_list = trades['actual_list']
#         target_list = trades['target_list']
#         trade_list = trades['trade_list']
#         # Trade masks.
#         # Used to identify open-close orders in both directions.
#         mOS = (target_list < 0) * (trade_list < 0)
#         mOL = (target_list > 0) * (trade_list > 0)
#         mCS = (actual_list < 0) * (trade_list > 0)
#         mCL = (actual_list > 0) * (trade_list < 0)
#         # Trades.
#         # Identified trades in both directions.
#         short_open = np.abs(target_list * mOS - actual_list * mOS * (actual_list < 0))
#         short_close = np.abs(trade_list * mCS - target_list * mCS * (target_list > 0))
#         long_open = np.abs(target_list * mOL - actual_list * mOL * (actual_list > 0))
#         long_close = np.abs(trade_list * mCL - target_list * mCL * (target_list < 0))
#         # Trades current margin.
#         margin_open = np.sum((long_open + short_open) * price * portfolio.regT_margin)
#         margin_close = np.sum((long_close - short_close) * price * portfolio.regT_margin)
#         # RegT current margin.
#         regT_current = margin_close - margin_open
#         return regT_current
#
#     # Fn: (11)
#     # Available Funds.
#     @staticmethod
#     def get_available_funds(equity, initial):
#         available = equity - initial
#         return available
#
#     # Fn: (12)
#     # Excess Liquidity.
#     @staticmethod
#     def get_excess_liquidity(equity, maintenance):
#         excess = equity - maintenance
#         return excess
#
#     # Fn: (13)
#     # Special Memorandum Account.
#     @staticmethod
#     def get_sma(portfolio, equity, regT_total, regT_current, day, divs, comm):
#         sma1 = portfolio.funds['sma'][day] + regT_current + divs - comm
#         sma2 = equity - regT_total
#         sma = np.maximum(sma1, sma2)
#         return sma
#
#     # Fn: (14)
#     # Buying Power.
#     @staticmethod
#     def get_buying_power(portfolio, sma, excess):
#         bp = np.minimum(sma / portfolio.initial_margin, excess / portfolio.maintenance_margin)
#         return bp
#
#     # Fn: (15)
#     # Leverage.
#     @staticmethod
#     def get_leverage(equity, longs, shorts):
#         leverage = hl.nan_to_zero(equity / (longs + shorts))
#         return leverage
#
#     # Fn: (16)
#     # Price Slippage.
#     @staticmethod
#     def get_price_slippage(portfolio, price, day):
#         price_slippage = np.zeros_like(price, dtype=np.float32)
#         return price_slippage
#
#     # Fn: (17)
#     # Trade Commission.
#     @staticmethod
#     def get_commission(commissions, positions, positions_value):
#         # Compute trade commission based
#         # on comparing trade size vs min-max
#         # trade commissions set up by brokerage firm.
#         trades = np.copy(np.abs(positions))                                            # trades (in units)
#         trades_value = np.copy(np.abs(positions_value))                                # trades (in USD)
#         commission = trades * commissions['size']                                      # general commission per trade
#         commission = np.maximum(commission, commissions['min'])                        # minimum constraint applied
#         commission = np.minimum(commission, (trades_value * commissions['max']))       # maximum constraint applied
#         commission = commission * (trades > 0)                                         # relevant trades only
#         commission = hl.zero_to_nan(commission)                                        # ....
#         commission = hl.nan_to_zero(commission)                                        # ....
#         commission = np.sum(commission)                                                # total commission
#         return commission
#
# if __name__ == '__main__':
#     data = joblib.load('data/pp_data/data_dict/data.pickle')
#     first_index = data['first.index']
#     positions_strategy = joblib.load('data/pp_data/positions/positions.pickle')
#     strategy = Simulation(positions_strategy).run_simulation()
