from __future__ import print_function
import pandas as pd
import numpy as np
import configparser
import warnings
import os
import locale
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from sklearn.linear_model import LinearRegression
from simulation_fn import Simulation
from sklearn.externals import joblib
pd.set_option('display.max_colwidth', 1500)
pd.set_option('display.notebook_repr_html', True)
pd.set_option('display.width', 900)
locale.setlocale(locale.LC_ALL, 'English_United States.1252')
warnings.filterwarnings('ignore')


class Evaluation():

    """
    Show strategy performance.
    : input: sim object.
    : return: plot results
    : store: save report.xlsx (data/pp_data/res_data/)

    Brief description:
        - Show results and basic performance metrics compared to SP500.

    """

    def __init__(self, sim_strategy):

        # Path to options file.
        path_options = 'options.ini'

        # Load config file.
        config = configparser.ConfigParser()                                            # define config object.
        config.optionxform = str                                                        # hold keys register
        config.read(path_options)                                                       # read config file.

        # Path variables.
        self.path_data = config['PATH'].get('DataObj')                                  # address: data object
        self.path_report = config['PATH'].get('ReportObj')                              # address: results report
        self.path_spx_price = config['PATH'].get('SPXDataPrice')                        # address: spx price

        # Load data object.
        self.data = joblib.load(self.path_data)

        # Params.
        self.first_index = self.data['first.index']                                     # sim start index
        self.initial_capital = config['SIMULATION'].getfloat('InitialCapital')          # initial capital

        # Data objects
        self.sim_strategy = sim_strategy                                                # simulated strategy results

    # Fn: (1)
    # Analyze portfolio simulation results.
    def run_evaluation(self, save_report=True):

        print('')
        print('Evaluation started:')

        # Convert dates from int to time format
        dates = pd.to_datetime(self.data['dates'].astype(np.str), format='%Y%m%d')
        dates = dates[self.first_index:-1]

        # Load simulation data.
        strategy_portfolio = self.sim_strategy['portfolio']
        strategy_trade_cost = self.sim_strategy['trade_cost']

        # Load SP500 benchmark.
        spx_data = pd.read_csv(self.path_spx_price, header=0)
        spx_returns = spx_data['Close'].pct_change().values[self.first_index:-1]
        spx_returns[0] = 0.00
        spx_returns = np.cumprod(np.add(spx_returns, 1), axis=0)
        market_equity = np.multiply(spx_returns, self.initial_capital)
        market_equity[0] = self.initial_capital

        # Load all simulation vars.
        strategy_equity = strategy_portfolio['equity']
        strategy_cash = strategy_portfolio['cash']
        strategy_value = strategy_portfolio['value'].transpose()
        strategy_pnl = strategy_portfolio['pnl']
        strategy_turn = strategy_portfolio['turnover']
        year_idx = np.arange(0, (len(dates) + 252), 252)
        # Trade cost.
        ex_com = strategy_trade_cost['exchange_commission']
        price_slip = strategy_trade_cost['price_slippage']

        # Strategy performance metrics.
        # -----------------------------------
        s_ret_total = ((strategy_equity[-1] - strategy_equity[0]) / strategy_equity[0]) * 100
        s_returns = pd.DataFrame(strategy_equity).pct_change().values.ravel()[1:]
        s_mean_ret = 0
        s_mean_ir = 0
        for idx, y_idx in enumerate(year_idx[:-1]):
            start_idx = y_idx
            end_idx = year_idx[idx+1]
            start_balance = strategy_equity[start_idx:end_idx][0]
            end_balance = strategy_equity[start_idx:end_idx][-1]
            s_ret_arr = pd.DataFrame(strategy_equity[start_idx:end_idx]).pct_change().values.ravel()[1:]
            s_ret = (end_balance - start_balance) / start_balance
            s_ir = (np.mean(s_ret_arr) / np.std(s_ret_arr)) * np.sqrt(len(s_ret_arr)+1)
            s_mean_ret += s_ret
            s_mean_ir += s_ir
        s_mean_ret = (s_mean_ret / (len(dates) / 252)) * 100
        s_mean_ir = (s_mean_ir / (len(dates) / 252))
        s_mdd, s_mdd_days = self.mdd(s_returns)
        s_mdd = s_mdd * 100
        s_mdd_days = s_mdd_days[1] - s_mdd_days[0]
        turnover_avr = np.mean(strategy_turn) * 100
        trade_cost_total = locale.currency(np.sum(ex_com + price_slip), grouping=True)

        # Market performance metrics.
        # -----------------------------------
        m_ret_total = ((market_equity[-1] - market_equity[0]) / market_equity[0]) * 100
        m_returns = pd.DataFrame(market_equity).pct_change().values.ravel()[1:]
        m_mean_ret = 0
        m_mean_ir = 0
        for idx, y_idx in enumerate(year_idx[:-1]):
            start_idx = y_idx
            end_idx = year_idx[idx + 1]
            start_balance = market_equity[start_idx:end_idx][0]
            end_balance = market_equity[start_idx:end_idx][-1]
            m_ret_arr = pd.DataFrame(market_equity[start_idx:end_idx]).pct_change().values.ravel()[1:]
            m_ret = (end_balance - start_balance) / start_balance
            m_ir = (np.mean(m_ret_arr) / np.std(m_ret_arr)) * np.sqrt(len(m_ret_arr) + 1)
            m_mean_ret += m_ret
            m_mean_ir += m_ir
        m_mean_ret = (m_mean_ret / (len(dates) / 252)) * 100
        m_mean_ir = (m_mean_ir / (len(dates) / 252))
        m_mdd, m_mdd_days = self.mdd(m_returns)
        m_mdd = m_mdd * 100
        m_mdd_days = m_mdd_days[1] - m_mdd_days[0]

        # Correlation and beta.
        corr = np.corrcoef(s_returns, m_returns)[0][1]
        beta = LinearRegression().fit(m_returns.reshape(-1, 1), s_returns.reshape(-1, 1)).coef_[0][0]

        # PLOT
        # -----------------------------------------------------------------------------
        # Plot metrics and compare simulation results against even weighted portfolio.
        plt.figure(figsize=(14, 10), frameon=False, facecolor='blue', edgecolor='orange')
        # Results
        ax1 = plt.subplot(2, 1, 1)
        plt.title('Results \n (correlation: {:.2f}. beta: {:.2f})'.format(corr, beta))
        plt.ylabel('Balance')
        plt.xlabel('Strategy:  r.total: {:.2f}%,  r.mean: {:.2f}%, '
                   ' ir mean: {:.2f},  dd: {:.2f}% ({} days) \n'
                   'Market:  r.total: {:.2f}%,  r.mean: {:.2f}%, '
                   ' ir mean: {:.2f},  dd: {:.2f}% ({} days) '.format(s_ret_total, s_mean_ret,
                                                                      s_mean_ir, s_mdd, s_mdd_days,
                                                                      m_ret_total, m_mean_ret, m_mean_ir,
                                                                      m_mdd, m_mdd_days))
        ax1.plot(dates, strategy_equity, color='blue', alpha=0.7, linestyle='solid')
        ax1.plot(dates, market_equity, color='red', alpha=0.7, linestyle='solid')
        ax1.yaxis.set_major_formatter(tkr.FuncFormatter(self.y_format_balance))
        ax1.legend(['strategy', 'market'], loc='upper left')
        ax1.grid(True)
        # Turnover
        ax2 = plt.subplot(2, 1, 2)
        plt.ylabel('PnL')
        plt.xlabel('Gross PnL: {:,.2f}. Turnover average: {:.2f} %. Total trade cost: {}'.
                   format(np.sum(strategy_portfolio['pnl']), turnover_avr, trade_cost_total))
        strategy_pnl[strategy_pnl > 15000] = 15000
        strategy_pnl[strategy_pnl < -15000] = -15000
        ax2.bar(dates, strategy_pnl * (strategy_pnl > 0), color='blue', alpha=0.6)
        ax2.bar(dates, strategy_pnl * (strategy_pnl < 0), color='red', alpha=0.6)
        ax2.grid(True)

        # SAVE
        # ------------------------------------------
        if save_report:
            # Create report folder is it doesn`t exist.
            if not os.path.exists(self.path_report.replace('report.xlsx', '')):
                os.makedirs(self.path_report.replace('report.xlsx', ''))
            # Save all data to excel file for further analysis
            writer = pd.ExcelWriter(self.path_report)
            # Write to file 'balance'
            strategy_equity = pd.DataFrame(strategy_equity, columns={'Equity'}, index=dates)
            market_equity = pd.DataFrame(market_equity, columns={'Market'}, index=dates)
            strategy_cash = pd.DataFrame(strategy_cash, columns={'Cash'}, index=dates)
            strategy_value = pd.DataFrame(np.sum(strategy_value, axis=1), columns={'Equity value'}, index=dates)
            # Write to file 'trade cost'
            ex_com = pd.DataFrame(np.cumsum(ex_com), columns={'Commission'}, index=dates)
            price_slip = pd.DataFrame(np.cumsum(price_slip), columns={'Slippage'}, index=dates)
            # Save data to excel sheet.
            data_to_save = pd.concat([strategy_value, strategy_cash, strategy_equity], axis=1)
            data_to_save.to_excel(writer, header=True, index=True, sheet_name='Portfolio')
            # Save data to excel sheet.
            data_to_save = pd.concat([strategy_value, market_equity, ex_com, price_slip], axis=1)
            data_to_save.to_excel(writer, header=True, index=True, sheet_name='Other')
            # Save excel file
            writer.save()
            print('  report saved')

        print('  evaluation completed')
        plt.show()

    # Fn: (2)
    # Max drawdown function.
    @staticmethod
    def mdd(returns, rolling=None):
        r = pd.DataFrame(returns).add(1).cumprod()
        x = pd.DataFrame(r.dot(1 / r.T.copy()) - 1)
        x.columns.name = 'start'
        x.index.name = 'end'
        y = x.stack().reset_index()
        y = y[y.start < y.end]
        if rolling is not None:
            y = y[y.end - y.start <= rolling]
        z = y.set_index(['start', 'end']).iloc[:, 0]
        return z.min(), z.argmin()

    # Fn: (3)
    # Function adds comma to tick values.
    @staticmethod
    def y_format_balance(x, pos):
        s = '%d' % x
        groups = []
        while s and s[-1].isdigit():
            groups.append(s[-3:])
            s = s[:-3]
        return s + ','.join(reversed(groups))

if __name__ == '__main__':

    data = joblib.load('data/pp_data/data_dict/data.pickle')
    first_index = data['first.index']
    positions_strategy = joblib.load('data/pp_data/positions/positions.pickle')
    strategy = Simulation(positions_strategy).run_simulation()
    Evaluation(strategy).run_evaluation(save_report=False)
