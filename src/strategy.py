import matplotlib.pyplot as plt
import backtrader as bt
import pandas as pd

OFF_BLACK = '#1e1e1e'
DARK_LABEL = '#808080'

plt.rcParams['figure.facecolor'] = OFF_BLACK
plt.rcParams['axes.facecolor'] = OFF_BLACK
plt.rcParams['savefig.facecolor'] = OFF_BLACK
plt.rcParams['text.color'] = DARK_LABEL
plt.rcParams['axes.labelcolor'] = DARK_LABEL
plt.rcParams['xtick.color'] = DARK_LABEL
plt.rcParams['ytick.color'] = DARK_LABEL
plt.rcParams['axes.edgecolor'] = DARK_LABEL


class BuyAndHold(bt.Strategy):
    def start(self):
        self.val_start = self.broker.get_cash()

    def next(self):
        if not self.position:
            size = int(self.broker.get_cash() / self.data.close[0])
            self.buy(size=size)


class DollarCostAverageDaily(bt.Strategy):
    params = (('cash_pct_per_trade', 0.05),)

    def __init__(self):
        self.last_day = None

    def next(self):
        current_day = self.data.datetime.date(0)

        if self.last_day is not None and current_day > self.last_day:
            cash = self.broker.get_cash()
            target_spend = cash * self.p.cash_pct_per_trade
            size = int(target_spend / self.data.close[0])
            if size > 0:
                self.buy(size=size)

        self.last_day = current_day


class ActiveTechAnalysis(bt.Strategy):
    params = (('ema_period', 5), ('rsi_period', 14))

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)
        # self.crossover_ema = bt.indicators.CrossOver(self.data.close, self.ema)

    def next(self):
        if len(self) < max(self.p.ema_period, self.p.rsi_period):
            return

        cash = self.broker.get_cash()
        size = int((cash * 0.95) / self.data.close[0])

        # --- Entry Logic ---
        if not self.position:
            # BUY if price is above EMA (uptrend) OR RSI is starting to move up from oversold (RSI < 40)
            if self.data.close[0] > self.ema[0] or self.rsi[0] < 40:
                if size > 0:
                    self.buy(size=size)

        # --- Exit Logic ---
        else:
            # SELL if price is below EMA (downtrend) OR RSI is overbought (RSI > 60)
            if self.data.close[0] < self.ema[0] or self.rsi[0] > 60:
                self.close()


def run_and_plot(strategy_class, data, name, results_dir='results', **kwargs):
    """Run a baseline strategy and save the backtest chart.

    Parameters
    ----------
    strategy_class : Backtrader Strategy class
    data           : Backtrader data feed
    name           : Human-readable strategy name used in the chart title and
                     output filename.
    results_dir    : Directory where the PNG will be saved (default: 'results').
    **kwargs       : Additional parameters forwarded to the Strategy.
    """
    print(f"--- Running: {name} ---")
    cerebro = bt.Cerebro()
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class, **kwargs)
    cerebro.broker.setcash(100000.0)

    start_value = 100000.0
    end_value = cerebro.broker.getvalue()
    profit_pct = ((end_value - start_value) / start_value) * 100

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Minutes, compression=60)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Backtest
    results = cerebro.run()
    strat = results[0]

    # Metrics
    start_value = 100000.0
    end_value = cerebro.broker.getvalue()
    profit_pct = ((end_value - start_value) / start_value) * 100

    # Sharpe
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    sharpe_print = f"{sharpe:.2f}" if sharpe is not None else "N/A"

    # Max Drawdown
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']

    print(f"Starting Cash:    ${start_value:,.2f}")
    print(f"Final Value:      ${end_value:,.2f}")
    print(f"Profit:           {profit_pct:.2f}%")
    print(f"Sharpe Ratio:     {sharpe_print}")
    print(f"Max Drawdown:     {max_dd:.2f}%")
    print("-" * 24)

    figures = cerebro.plot(
        iplot=False,
        style='candlestick',
        barup='green',
        bardown='red',
        volup='#00FF00',
        voldown='#FF0000',
        grid=False,
        loc=DARK_LABEL,
    )

    if len(figures) > 0 and len(figures[0]) > 0:
        fig = figures[0][0]

        fig.set_size_inches(18, 10)

        for ax in fig.get_axes():
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')

        fig.savefig(f'{results_dir}/Baseline: {name}-backtest.png')


class SentimentPandasData(bt.feeds.PandasData):
    lines = ('score_finbert_norm',)
    params = (('score_finbert_norm', -1), )


class SentimentStrategy(bt.Strategy):
    params = (('buy_thresh', 0.2), ('sell_thresh', -0.2))

    def next(self):
        sentiment = self.data.score_finbert_norm[0]

        if not self.position:
            if sentiment > self.p.buy_thresh:
                self.buy(size=100)
        else:
            if sentiment < self.p.sell_thresh:
                self.close()


class SentimentTAStrategy(bt.Strategy):
    params = (('ema_period', 20), ('rsi_period', 14))

    def __init__(self):
        self.ema = bt.indicators.EMA(self.data.close, period=self.p.ema_period)
        self.rsi = bt.indicators.RSI(self.data.close, period=self.p.rsi_period)

    def next(self):
        sentiment = self.data.score_finbert_norm[0]

        if not self.position:
            # Entry: Trend is up (Price > EMA) AND Sentiment is bullish
            if self.data.close[0] > self.ema[0] and sentiment > 0.1:
                self.buy(size=100)
        else:
            # Exit: RSI Overbought OR Sentiment turns bearish
            if self.rsi[0] > 70 or sentiment < -0.1:
                self.close()


class InverseSentimentStrategy(bt.Strategy):
    params = (('panic_thresh', -0.7), ('euphoria_thresh', 0.7))

    def next(self):
        sentiment = self.data.score_finbert_norm[0]

        if not self.position:
            # Buy the dip/panic
            if sentiment < self.p.panic_thresh:
                self.buy(size=100)
        else:
            # Sell the rally/euphoria
            if sentiment > self.p.euphoria_thresh:
                self.close()


def run_strategy(strategy_class, dataframe, name, results_dir='results', **kwargs):
    """Run a sentiment strategy and save the backtest chart.

    Parameters
    ----------
    strategy_class : Backtrader Strategy class
    dataframe      : pandas DataFrame with price and sentiment columns
    name           : Human-readable strategy name used in the chart title and
                     output filename.
    results_dir    : Directory where the PNG will be saved (default: 'results').
    **kwargs       : Additional parameters forwarded to the Strategy.
    """
    print(f"\n{'='*10} {name} {'='*10}")

    cerebro = bt.Cerebro()

    if not isinstance(dataframe.index, pd.DatetimeIndex):
        dataframe.index = pd.to_datetime(dataframe.index)

    data = SentimentPandasData(dataname=dataframe)
    cerebro.adddata(data)
    cerebro.addstrategy(strategy_class, **kwargs)
    cerebro.broker.setcash(100000.0)

    # Analyzers
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='sharpe',
                        timeframe=bt.TimeFrame.Minutes, compression=60)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

    # Backtest
    results = cerebro.run()
    strat = results[0]

    # Metrics
    start_val = 100000.0
    end_val = cerebro.broker.getvalue()
    profit_pct = ((end_val - start_val) / start_val) * 100

    # Sharpe
    sharpe = strat.analyzers.sharpe.get_analysis().get('sharperatio')
    sharpe_print = f"{sharpe:.2f}" if sharpe is not None else "N/A"

    # Max Drawdown
    max_dd = strat.analyzers.drawdown.get_analysis()['max']['drawdown']

    print(f"Final Value:      ${end_val:,.2f}")
    print(f"Profit:           {profit_pct:.2f}%")
    print(f"Sharpe Ratio:     {sharpe_print}")
    print(f"Max Drawdown:     {max_dd:.2f}%")

    figures = cerebro.plot(
        iplot=False,
        style='candlestick',
        barup='green', bardown='red',
        volup='#00FF00', voldown='#FF0000',
        grid=False,
        loc=DARK_LABEL
    )

    if len(figures) > 0 and len(figures[0]) > 0:
        fig = figures[0][0]
        fig.set_size_inches(18, 10)

        fig.savefig(f'{results_dir}/{name}-backtest.png')
