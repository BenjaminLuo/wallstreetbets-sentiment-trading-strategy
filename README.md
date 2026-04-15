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
├── data/
│   ├── reddit/                   # Raw daily WSB comment CSVs (gitignored – large)
│   ├── spy/                      # Intraday SPY price CSVs (gitignored – large)
│   ├── full_manual_spy.csv       # Cleaned, merged SPY data
│   ├── manual_merged_df.csv      # Merged sentiment + price data
│   ├── manual_spy_df.csv         # Processed SPY price data
│   └── manual_wsb_sdf.csv        # Processed WSB sentiment data (gitignored)
│
├── results/
│   ├── bear/                     # Backtest charts – bear market regime
│   ├── bull/                     # Backtest charts – bull market regime
│   ├── flat/                     # Backtest charts – flat/sideways regime
│   └── full/                     # Backtest charts – full period
│       ├── Baseline: Buy and Hold-backtest.png
│       ├── Baseline: Dollar Cost Averaging-backtest.png
│       ├── Baseline: Technical Analysis (EMA, BB, RSI, VWAP)-backtest.png
│       ├── Strategy 1: Pure Sentiment-backtest.png
│       ├── Strategy 2: Sentiment + Technical Analysis-backtest.png
│       └── Strategy 3: Inverse Sentiment (Contrarian)-backtest.png
│
├── utils/
│   ├── _processing.py            # NLP pipeline (cleaning, entity extraction, sentiment scoring)
│   └── _strategy.py              # Backtrader strategy classes + runner helpers
│
├── reddit_data.ipynb             # Data collection: PRAW scraping & preprocessing
├── intraday_spy_data.ipynb       # Data collection: SPY intraday price fetching
├── main.ipynb                    # End-to-end: merge, analyse, backtest, visualise
├── algo.ipynb                    # Experimental algorithmic trading sandbox
│
├── Final Presentation.pdf        # Project final report
├── Midterm Presentation.pdf      # Project midterm report
│
├── requirements.txt
├── .env.example                  # ← Copy to .env and fill in credentials
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

### 4. Configure Reddit API credentials

```bash
cp .env.example .env
# Edit .env with your CLIENT_ID, CLIENT_SECRET, and DEV_NAME
```

Register a Reddit app at https://www.reddit.com/prefs/apps (choose **script** type).

### 5. Run the notebooks

| Notebook | Purpose |
|----------|---------|
| `reddit_data.ipynb` | Scrape and preprocess r/WSB comments |
| `intraday_spy_data.ipynb` | Fetch SPY intraday OHLCV data |
| `main.ipynb` | Run the full analysis pipeline and backtests |
| `algo.ipynb` | Experimental trading logic |

---

## Results Snapshot

Backtest charts for all six strategies across the full evaluation period are saved in `results/full/`. Segmented results by market regime (bull, bear, flat) are in their respective subdirectories.

---

## Dependencies

See [`requirements.txt`](requirements.txt) for the full list. Key packages:

- **NLP:** `nltk`, `spacy`, `transformers` (FinBERT via HuggingFace)
- **Data:** `pandas`, `numpy`, `praw`
- **Backtesting:** `backtrader`
- **Visualisation:** `matplotlib`, `wordcloud`

---

## Presentations

- [`Final Presentation.pdf`](Final%20Presentation.pdf) — full project write-up and results
- [`Midterm Presentation.pdf`](Midterm%20Presentation.pdf) — intermediate progress report

---

## License

This project is for educational and research purposes only. **Nothing here constitutes financial advice.**
