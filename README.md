# WallStreetBets Sentiment Trading Strategy

> Backtesting a retail sentiment-driven trading strategy against SPY using Reddit r/WallStreetBets data and an ensemble NLP pipeline (VADER · Custom Lexicon · FinBERT).

---

## Overview

This project investigates whether crowd sentiment on **r/WallStreetBets** contains exploitable signal for short-term SPY trading. The pipeline:

1. **Collects** r/WSB comments via the Reddit PRAW API.
2. **Cleans & enriches** text (entity extraction, ticker detection).
3. **Scores sentiment** using a 4-model ensemble:
   - VADER (rule-based NLP)
   - Simple keyword lexicon
   - Weighted financial lexicon
   - FinBERT (ProsusAI/finbert, transformer-based)
4. **Merges** sentiment signals with intraday SPY OHLCV data.
5. **Backtests** three sentiment-driven strategies against three baselines using Backtrader.

---

## Repository Structure

```
wallstreetbets-sentiment-trading-strategy/
│
├── docs/
│   ├── Final Presentation.pdf      # Full project write-up & results
│   └── Midterm Presentation.pdf    # Intermediate progress report
│
├── notebooks/
│   ├── 01_data_collection.ipynb    # Reddit scraper (PRAW) + SPY fetcher (yfinance) – optional
│   ├── 02_main.ipynb               # End-to-end: merge → analyse → backtest
│   └── 03_algo.ipynb               # Experimental algorithmic trading sandbox
│
├── src/                            # Python source package
│   ├── __init__.py
│   ├── processing.py               # NLP pipeline (cleaning, entity extraction,
│   │                               #   sentiment scoring, visualisation helpers)
│   └── strategy.py                 # Backtrader strategy classes + runner helpers
│
├── data/
│   ├── full_manual_spy.csv         # ✅ committed – merged SPY price data
│   ├── manual_merged_df.csv        # ✅ committed – merged sentiment + price data
│   ├── manual_spy_df.csv           # ✅ committed – processed SPY price data
│   ├── reddit/                     # raw daily WSB comment CSVs  (gitignored – re-scrape if needed)
│   └── spy/                        # raw intraday SPY CSVs       (gitignored – re-fetch if needed)
│
├── results/
│   ├── bear/                       # Backtest charts – bear market regime
│   ├── bull/                       # Backtest charts – bull market regime
│   ├── flat/                       # Backtest charts – flat/sideways regime
│   └── full/                       # Backtest charts – full evaluation period
│
├── requirements.txt
├── .env.example                    # ← Copy to .env and fill in credentials
└── .gitignore
```

---

## Strategies Tested

| # | Strategy | Description |
|---|----------|-------------|
| B1 | **Buy & Hold** | Buy once, hold for the full period |
| B2 | **Dollar Cost Averaging** | Invest 5% of cash every trading day |
| B3 | **Technical Analysis** | EMA-5 + RSI-14 crossover signals |
| S1 | **Pure Sentiment** | Buy when FinBERT score > 0.2; sell < -0.2 |
| S2 | **Sentiment + TA** | Buy when price > EMA-20 AND sentiment > 0.1 |
| S3 | **Inverse Sentiment (Contrarian)** | Fade the crowd: buy on panic, sell on euphoria |

---

## Sentiment Ensemble

The ensemble score is the mean of four independent signals:

```
ensemble = (VADER + Lexicon + FinLex + FinBERT) / 4
```

| Model | Approach | Handles WallStreetBets Slang |
|-------|----------|------------------------------|
| VADER | Rule-based polarity | ✓ (general) |
| Keyword Lexicon | Bullish/bearish word sets | ✓ (custom) |
| Financial Lexicon | Weighted domain vocabulary | ✓✓ (WSB-specific words) |
| FinBERT | Transformer (ProsusAI) | ✓✓✓ (fine-tuned on financial text) |

---

## Quickstart

### 1. Clone the repo

```bash
git clone https://github.com/<your-username>/wallstreetbets-sentiment-trading-strategy.git
cd wallstreetbets-sentiment-trading-strategy
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

> **Apple Silicon / CUDA users:** Install the appropriate PyTorch wheel first:
> https://pytorch.org/get-started/locally/

### 4. Configure Reddit API credentials *(optional – only needed to re-scrape data)*

```bash
cp .env.example .env
# Edit .env with your CLIENT_ID, CLIENT_SECRET, and DEV_NAME
```

Register a Reddit app at https://www.reddit.com/prefs/apps (choose **script** type).  
Skip this step if you are using the pre-collected data already committed to `data/`.

### 5. Run the notebooks

| Notebook | Purpose | Required? |
|----------|---------|:---------:|
| `notebooks/01_data_collection.ipynb` | Re-scrape WSB comments (PRAW) and/or re-fetch SPY bars (yfinance) | Optional |
| `notebooks/02_main.ipynb` | Full analysis pipeline and backtests | **Yes** |
| `notebooks/03_algo.ipynb` | Experimental algorithmic trading sandbox | Optional |

> **TL;DR:** Processed data (`full_manual_spy.csv`, `manual_merged_df.csv`, `manual_spy_df.csv`) is already
> committed to `data/`. You can jump straight to **`notebooks/02_main.ipynb`** without running
> the data-collection notebook or configuring any API credentials.

---

## Results Snapshot

Backtest charts are saved under `results/<regime>/` for each of the four market regimes evaluated.  
The naming convention is:

| File pattern | Description |
|---|---|
| `Baseline: Buy and Hold-backtest.png` | Buy-and-hold benchmark |
| `Baseline: Dollar Cost Averaging-backtest.png` | DCA benchmark |
| `Baseline: Technical Analysis (EMA, BB, RSI, VWAP)-backtest.png` | TA benchmark |
| `Strategy 1: Pure Sentiment-backtest.png` | S1 – FinBERT score only |
| `Strategy 2: Sentiment + Technical Analysis-backtest.png` | S2 – FinBERT + EMA/RSI |
| `Strategy 3: Inverse Sentiment (Contrarian)-backtest.png` | S3 – Contrarian |

### Full Period

| Buy & Hold | Dollar Cost Averaging |
|---|---|
| ![Full – Buy and Hold](results/full/Baseline:%20Buy%20and%20Hold-backtest.png) | ![Full – DCA](results/full/Baseline:%20Dollar%20Cost%20Averaging-backtest.png) |

| Technical Analysis | Pure Sentiment |
|---|---|
| ![Full – TA](results/full/Baseline:%20Technical%20Analysis%20(EMA,%20BB,%20RSI,%20VWAP)-backtest.png) | ![Full – S1](results/full/Strategy%201:%20Pure%20Sentiment-backtest.png) |

| Sentiment + TA | Inverse Sentiment |
|---|---|
| ![Full – S2](results/full/Strategy%202:%20Sentiment%20+%20Technical%20Analysis-backtest.png) | ![Full – S3](results/full/Strategy%203:%20Inverse%20Sentiment%20(Contrarian)-backtest.png) |

### Bull Regime

| Buy & Hold | Dollar Cost Averaging |
|---|---|
| ![Bull – Buy and Hold](results/bull/Baseline:%20Buy%20and%20Hold-backtest.png) | ![Bull – DCA](results/bull/Baseline:%20Dollar%20Cost%20Averaging-backtest.png) |

| Technical Analysis | Pure Sentiment |
|---|---|
| ![Bull – TA](results/bull/Baseline:%20Technical%20Analysis%20(EMA,%20BB,%20RSI,%20VWAP)-backtest.png) | ![Bull – S1](results/bull/Strategy%201:%20Pure%20Sentiment-backtest.png) |

| Sentiment + TA | Inverse Sentiment |
|---|---|
| ![Bull – S2](results/bull/Strategy%202:%20Sentiment%20+%20Technical%20Analysis-backtest.png) | ![Bull – S3](results/bull/Strategy%203:%20Inverse%20Sentiment%20(Contrarian)-backtest.png) |

### Bear Regime

| Buy & Hold | Dollar Cost Averaging |
|---|---|
| ![Bear – Buy and Hold](results/bear/Baseline:%20Buy%20and%20Hold-backtest.png) | ![Bear – DCA](results/bear/Baseline:%20Dollar%20Cost%20Averaging-backtest.png) |

| Technical Analysis | Pure Sentiment |
|---|---|
| ![Bear – TA](results/bear/Baseline:%20Technical%20Analysis%20(EMA,%20BB,%20RSI,%20VWAP)-backtest.png) | ![Bear – S1](results/bear/Strategy%201:%20Pure%20Sentiment-backtest.png) |

| Sentiment + TA | Inverse Sentiment |
|---|---|
| ![Bear – S2](results/bear/Strategy%202:%20Sentiment%20+%20Technical%20Analysis-backtest.png) | ![Bear – S3](results/bear/Strategy%203:%20Inverse%20Sentiment%20(Contrarian)-backtest.png) |

### Flat/Sideways Regime

| Buy & Hold | Dollar Cost Averaging |
|---|---|
| ![Flat – Buy and Hold](results/flat/Baseline:%20Buy%20and%20Hold-backtest.png) | ![Flat – DCA](results/flat/Baseline:%20Dollar%20Cost%20Averaging-backtest.png) |

| Technical Analysis | Pure Sentiment |
|---|---|
| ![Flat – TA](results/flat/Baseline:%20Technical%20Analysis%20(EMA,%20BB,%20RSI,%20VWAP)-backtest.png) | ![Flat – S1](results/flat/Strategy%201:%20Pure%20Sentiment-backtest.png) |

| Sentiment + TA | Inverse Sentiment |
|---|---|
| ![Flat – S2](results/flat/Strategy%202:%20Sentiment%20+%20Technical%20Analysis-backtest.png) | ![Flat – S3](results/flat/Strategy%203:%20Inverse%20Sentiment%20(Contrarian)-backtest.png) |

> **Generating figures yourself**: run `notebooks/02_main.ipynb` end-to-end.  The notebooks call
> `run_and_plot(..., results_dir='../results/full')` and
> `run_strategy(..., results_dir='../results/full')` from `src/strategy.py`, and
> `plot_sentiment(df, ..., save_path='../results/sentiment_dist.png')` /
> `plot_sector_sentiment_trends(df, save_path='../results/sector_trends.png')` /
> `plot_sentiment_vs_price(df, save_path='../results/price_vs_sentiment.png')` from
> `src/processing.py`.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

- **NLP:** `nltk`, `spacy`, `transformers` (FinBERT via HuggingFace)
- **Data:** `pandas`, `numpy`, `praw`
- **Backtesting:** `backtrader`
- **Visualisation:** `matplotlib`, `wordcloud`

---

## Presentations

- [`docs/Final Presentation.pdf`](docs/Final%20Presentation.pdf) — full project write-up and results
- [`docs/Midterm Presentation.pdf`](docs/Midterm%20Presentation.pdf) — intermediate progress report

---

## License

This project is for educational and research purposes only. **Nothing here constitutes financial advice.**
