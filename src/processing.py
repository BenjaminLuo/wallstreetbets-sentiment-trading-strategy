"""Data processing functions"""

import re
import string
import spacy
import nltk
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from transformers import pipeline
from tqdm import tqdm
from wordcloud import WordCloud, STOPWORDS


stopwords.words('english')
tqdm.pandas()
nlp = spacy.load('en_core_web_lg')

SIA = SentimentIntensityAnalyzer()

FINANCIAL_SENTIMENT_LEXICON = {
    # Positive
    'profit': 0.9, 'gains': 0.8, 'soar': 0.8, 'rally': 0.6, 'up': 0.3,
    'rich': 0.8, 'winning': 0.7, 'green': 0.6, 'money': 0.4, 'call': 0.5,
    'calls': 0.5, 'long': 0.5, 'buy': 0.4, 'bull': 0.6, 'bulls': 0.6,
    'win': 0.6, 'printing': 0.7, 'moon': 0.9, 'rocket': 0.9, 'ath': 0.7,
    # Negative
    'loss': -0.9, 'crash': -0.9, 'dump': -0.7, 'red': -0.6, 'down': -0.3,
    'poor': -0.8, 'losing': -0.7, 'tank': -0.8, 'drill': -0.7, 'put': -0.5,
    'puts': -0.5, 'short': -0.5, 'sell': -0.4, 'bear': -0.6, 'bears': -0.6,
    'rekt': -0.9, 'drop': -0.4, 'plummet': -0.8, 'help': -0.4, 'anger': -0.5,
    # Neutral/Context
    'flat': 0.0, 'theta': 0.0, 'gamma': 0.0
}


# --------------------------------------------------------------------------
# FinBERT initialization ---------------------------------------------------

DEVICE = -1  # Default to CPU
if torch.cuda.is_available():
    DEVICE = 0  # CUDA
elif torch.backends.mps.is_available():
    DEVICE = "mps"  # Apple MPS

FINBERT = pipeline(
    "sentiment-analysis",
    model="ProsusAI/finbert",
    truncation=True,
    max_length=512,
    device=DEVICE
)


def clean_text(text):
    """Data cleaning"""
    text = text.lower()

    # Patterns
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'img\s*emote\s*t5\s*2th52\s*\d+', '', text)
    text = re.sub(r'img\s*emote\s*t5\s*\d+', '', text)
    text = re.sub(r'img\s*emote', '', text)
    exclude = set(string.punctuation.replace('!', '').replace('?', ''))
    text = ''.join(ch for ch in text if ch not in exclude)
    text = re.sub(r'\d+', '', text)

    # Stop words
    stop_words = set(stopwords.words('english'))
    wsb_noise = {'im', 'isnt', 'its', 'youre',
                 'theyre', 'dont', 'get', 'like', 'would', 'can'}
    stop_words.update(wsb_noise)
    word_tokens = nltk.word_tokenize(text)
    filtered_words = [w for w in word_tokens if not w in stop_words]

    text = ' '.join(filtered_words)
    text = re.sub(r'\s+', ' ', text).strip()

    return str(text)


def entity_extraction(text):
    """
    Extracts highly relevant financial entities (tickers, organizations, money)
    from text, prioritizing ALL-CAPS words that are 3-5 characters long.
    """

    ticker_regex = r'\b[A-Z]{3,5}\b'
    all_caps_tickers = re.findall(ticker_regex, text.upper())

    doc = nlp(text)
    relevant_labels = ('ORG', 'MONEY', 'PRODUCT', 'GPE')

    spacy_entities = [
        ent.text.upper() for ent in doc.ents
        if ent.label_ in relevant_labels
    ]

    all_entities = set(all_caps_tickers + spacy_entities)

    blacklist = {
        'IM', 'DONT', 'CAN', 'ALL', 'THE', 'AHEAD', 'TODAY', 'YESTERDAY',
        'LOL', 'WSB', 'BIGGEST', 'PUMP', 'DUMP', 'LOST', 'HAS', 'WHAT',
        'HAVE', 'THAN', 'SAME', 'PURE', 'LOGIC', 'WOULD', 'PRAYING', 'AWAY', 'IMG', 'EMOTE',
        'HEAD', 'SHOULDERS', 'WEEK', 'NOW', 'BUY', 'SELL', 'PCE'  # Add more as needed
    }

    final_entities = [
        e for e in all_entities
        if e not in blacklist and len(e) >= 3  # Ensure minimum length
    ]

    return ', '.join(sorted(list(set(final_entities))))


def ticker_extraction(text):
    """Extract, filter, and normalize ticker symbols."""
    whitelist = {"SPY", "ASTS", "RKLB", "GME", "AMC", "TSLA", "NVDA", "PLTR", "META"}
    blacklist = {"discord", "gg", "https", "com"}

    doc = nlp(str(text))
    extracted = [ent.text for ent in doc.ents if ent.label_ == "ORG"]

    # Whitelist
    words = set(str(text).split())
    found_whitelist = [word for word in words if word.upper() in whitelist]
    combined = extracted + found_whitelist

    # Processing
    processed_tickers = []
    for ticker in combined:
        clean_ticker = ticker.lower()
        if clean_ticker not in blacklist:
            processed_tickers.append(clean_ticker)

    unique_tickers = list(set(processed_tickers))

    return ", ".join(unique_tickers) if unique_tickers else None


def score_vader(text):
    """Method 1: VADER (Best for natural language context)"""
    if not text or SIA is None:
        return 0.0
    return SIA.polarity_scores(text)['compound']


def score_lexicon(text):
    """Method 2: Simple Keyword Lexicon (Expanded)"""
    if not text:
        return 0.0
    score = 0
    words = set(text.split())

    bullish = {
        'long', 'buy', 'calls', 'positive', 'win', 'winning', 'green',
        'rich', 'up', 'moon', 'rocket', 'print', 'bull', 'bulls'
    }
    bearish = {
        'short', 'sell', 'puts', 'negative', 'bear', 'bears', 'red',
        'loss', 'poor', 'down', 'crash', 'tank', 'drill', 'drop'
    }

    hits = 0
    for word in words:
        if word in bullish:
            score += 0.5
            hits += 1
        elif word in bearish:
            score -= 0.5
            hits += 1

    if hits == 0:
        return 0.0

    return np.clip(score, -1.0, 1.0)


def score_financial_lex(text):
    """Method 3: Weighted Financial Lexicon"""
    if not text:
        return 0.0
    score = 0
    words = text.split()
    match_count = 0

    for word in words:
        if word in FINANCIAL_SENTIMENT_LEXICON:
            score += FINANCIAL_SENTIMENT_LEXICON[word]
            match_count += 1

    if match_count == 0:
        return 0.0

    return np.clip(score / max(1, match_count), -1.0, 1.0)


def score_finbert_batch(texts, batch_size=32):
    """
    Method 4: FinBERT (Continuous Scoring)
    Returns: Probability(Positive) - Probability(Negative)
    """
    if FINBERT is None:
        return [0.0] * len(texts)

    scores = []

    try:
        iterator = tqdm(range(0, len(texts), batch_size), desc="FinBERT Processing")

        for i in iterator:
            batch = texts[i: i + batch_size]
            batch = [str(t).strip() if (t is not None and str(t).strip() != "") else "neutral" for t in batch]
            results = FINBERT(batch, batch_size=batch_size, top_k=None)

            for res_list in results:
                if isinstance(res_list, dict):
                    res_list = [res_list]

                pos_score = 0.0
                neg_score = 0.0

                for item in res_list:
                    if item['label'] == 'positive':
                        pos_score = item['score']
                    elif item['label'] == 'negative':
                        neg_score = item['score']

                continuous_score = pos_score - neg_score
                scores.append(continuous_score)

    except Exception as e:
        print(f"Error during FinBERT inference: {e}")
        remaining = len(texts) - len(scores)
        scores.extend([0.0] * remaining)

    return scores


def sentiment_analysis(df):
    """
    Applies all FOUR sentiment methods to the cleaned text and calculates the average.
    """
    if df.empty:
        print("Input DataFrame is empty.")
        return df

    print("--- Starting Ensemble Sentiment Calculation ---")

    # Helper for progress bar application
    def apply_func(series, func, desc="Processing"):
        tqdm.pandas(desc=desc)
        return series.progress_apply(func)

    # Apply cleaning
    source_col = 'text'
    print(f"Cleaning text from column: {source_col}")

    # 1. VADER
    print("Calculating VADER scores...")
    df['score_vader'] = apply_func(
        df['cleaned_text'], score_vader, desc="Scoring VADER")

    # 2. Basic Lexicon
    print("Calculating Lexicon scores...")
    df['score_lexicon'] = apply_func(
        df['cleaned_text'], score_lexicon, desc="Scoring Lexicon")

    # 3. Financial Lexicon
    print("Calculating Financial Lexicon scores...")
    df['score_financial_lex'] = apply_func(
        df['cleaned_text'], score_financial_lex, desc="Scoring FinLex")

    # 4. FinBERT (Batch Processed)
    if FINBERT:
        print(f"Running FinBERT on {len(df)} rows...")
        text_list = df['text'].tolist()
        df['score_finbert'] = score_finbert_batch(text_list, batch_size=32)

        # Average of 4 methods
        df['ensemble_sentiment'] = (
            df['score_vader'] +
            df['score_lexicon'] +
            df['score_financial_lex'] +
            df['score_finbert']
        ) / 4.0
    else:
        print("Skipping FinBERT (model not loaded).")
        df['score_finbert'] = 0.0
        # Average of 3 methods
        df['ensemble_sentiment'] = (
            df['score_vader'] +
            df['score_lexicon'] +
            df['score_financial_lex']
        ) / 3.0

    print("Ensemble scoring complete.")
    return df


def show_wordcloud(data, title="", save_path=None):
    """Displays a word cloud showing most popular tickers.

    Parameters
    ----------
    data      : pandas Series of ticker/text strings
    title     : chart title
    save_path : optional file path (e.g. 'results/wordcloud.png'); if supplied
                the figure is saved before being displayed.
    """
    text = " ".join(t for t in data.dropna())
    stopwords = set(STOPWORDS)
    stopwords.update(["t", "co", "https", "amp", "U", "fuck", "fucking"])
    wordcloud = WordCloud(stopwords=stopwords, scale=4, max_font_size=50,
                          max_words=500, background_color="black").generate(text)
    fig = plt.figure(1, figsize=(16, 16))
    plt.axis('off')
    fig.suptitle(title, fontsize=20)
    fig.subplots_adjust(top=2.3)
    plt.imshow(wordcloud, interpolation='bilinear')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def find_sentiment(post):
    """SIA approach for sentiment processing, used for word cloud generation"""
    if SIA.polarity_scores(post)["compound"] > 0:
        return 1
    elif SIA.polarity_scores(post)["compound"] < 0:
        return -1
    else:
        return 0


def plot_sentiment(df, feature, title, save_path=None):
    """For plotting the sentiment distribution of a given feature column.

    Parameters
    ----------
    df        : DataFrame containing *feature*
    feature   : column name whose value_counts will be plotted
    title     : chart title
    save_path : optional file path; if supplied the figure is saved to disk
                before being displayed (e.g. 'results/sentiment_dist.png').
    """
    counts = df[feature].value_counts()
    percent = counts/sum(counts)

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 5))

    counts.plot(kind='bar', ax=ax1, color='green')
    percent.plot(kind='bar', ax=ax2, color='blue')
    ax1.set_ylabel(f'Counts : {title} sentiments', size=12)
    ax2.set_ylabel(f'Percentage : {title} sentiments', size=12)
    plt.suptitle(f"Sentiment analysis: {title}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def plot_sector_sentiment_trends(df, rolling_window=24, save_path=None):
    """
    Cleans ticker data, maps to known sectors, and plots smoothed sentiment trends.

    Parameters
    ----------
    df             : The wsb_sdf dataframe.
    rolling_window : Number of hours for the moving average (default 24 h).
    save_path      : Optional file path; if supplied the figure is saved before
                     being displayed (e.g. 'results/sector_trends.png').
    """
    # 1. SETUP: Dark Theme & Muted Palette
    plt.style.use('dark_background')
    muted_colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F']

    # 2. DATA CLEANING: Explode and filter noisy tickers
    df_clean = df.copy()
    df_clean['ticker_list'] = df_clean['ticker'].str.split(',')
    df_exploded = df_clean.explode('ticker_list')
    df_exploded['clean_ticker'] = df_exploded['ticker_list'].str.strip().str.upper()

    # 3. SECTOR MAPPING: Reliable static mapping to avoid API 404s
    sector_map = {
        # 1. MAG 7 & BIG TECH (The Market Drivers)
        'AAPL': 'Mag7/Big Tech', 'APPLE': 'Mag7/Big Tech', 'MSFT': 'Mag7/Big Tech', 'MICROSOFT': 'Mag7/Big Tech',
        'GOOGL': 'Mag7/Big Tech', 'GOOG': 'Mag7/Big Tech', 'GOOGLE': 'Mag7/Big Tech', 'FB': 'Mag7/Big Tech',
        'META': 'Mag7/Big Tech', 'FACEBOOK': 'Mag7/Big Tech', 'AMZN': 'Mag7/Big Tech', 'AMAZON': 'Mag7/Big Tech',
        'TSLA': 'Mag7/Big Tech', 'TESLA': 'Mag7/Big Tech', 'NFLX': 'Mag7/Big Tech', 'NETFLIX': 'Mag7/Big Tech',

        # 2. SEMICONDUCTORS & AI (The Infrastructure)
        'NVDA': 'Semis/AI Infrastructure', 'NVIDIA': 'Semis/AI Infrastructure', 'NVDIA': 'Semis/AI Infrastructure',
        'AMD': 'Semis/AI Infrastructure', 'AVGO': 'Semis/AI Infrastructure', 'SMCI': 'Semis/AI Infrastructure',
        'INTC': 'Semis/AI Infrastructure', 'INTEL': 'Semis/AI Infrastructure', 'MU': 'Semis/AI Infrastructure',
        'PLTR': 'Semis/AI Infrastructure', 'PALANTIR': 'Semis/AI Infrastructure', 'ARM': 'Semis/AI Infrastructure',

        # 3. FRONTIER TECH (Space, Quantum, & Nuclear Energy)
        'RKLB': 'Frontier (Space/Quantum/Nuclear)', 'ASTS': 'Frontier (Space/Quantum/Nuclear)',
        'RGTI': 'Frontier (Space/Quantum/Nuclear)', 'IONQ': 'Frontier (Space/Quantum/Nuclear)',
        'OKLO': 'Frontier (Space/Quantum/Nuclear)', 'SMR': 'Frontier (Space/Quantum/Nuclear)',
        'LUNR': 'Frontier (Space/Quantum/Nuclear)', 'NNE': 'Frontier (Space/Quantum/Nuclear)',

        # 4. MEME & RETAIL HIGH-BETA (The Gambles)
        'GME': 'Meme/High-Beta Retail', 'GAMESTOP': 'Meme/High-Beta Retail', 'AMC': 'Meme/High-Beta Retail',
        'DJT': 'Meme/High-Beta Retail', 'MSTR': 'Meme/High-Beta Retail', 'COIN': 'Meme/High-Beta Retail',
        'HOOD': 'Meme/High-Beta Retail', 'ROBINHOOD': 'Meme/High-Beta Retail', 'RDDT': 'Meme/High-Beta Retail',

        # 5. FINANCIALS & SERVICES (Traditional Players)
        'JPM': 'Financials/Services', 'CHASE': 'Financials/Services', 'GS': 'Financials/Services',
        'VISA': 'Financials/Services', 'MA': 'Financials/Services', 'PYPL': 'Financials/Services',
        'WMT': 'Financials/Services', 'WALMART': 'Financials/Services', 'COST': 'Financials/Services',

        # 6. MACRO & INDICES (The Benchmarks)
        'SPY': 'Macro/Indices', 'QQQ': 'Macro/Indices', 'IWM': 'Macro/Indices', 'SPX': 'Macro/Indices',
        'VIX': 'Macro/Indices', 'BTC': 'Macro/Indices', 'BITCOIN': 'Macro/Indices', 'GLD': 'Macro/Indices'
    }
    df_exploded['sector'] = df_exploded['clean_ticker'].map(sector_map)

    # Filter out "Unknown" results to focus on real market trends
    df_filtered = df_exploded.dropna(subset=['sector']).copy()

    # 4. TIME-SERIES PROCESSING
    df_filtered['datetime'] = pd.to_datetime(df_filtered['datetime'])
    df_time = df_filtered.set_index('datetime')

    # Resample to hourly ('h') to resolve FutureWarning
    # Reindexing ensures lines don't "jump" across gaps in the timeline
    sector_sentiment_trend = df_time.groupby(['sector', pd.Grouper(freq='h')])['ensemble_sentiment'].mean().unstack(level=0)
    all_hours = pd.date_range(start=sector_sentiment_trend.index.min(), 
                              end=sector_sentiment_trend.index.max(), 
                              freq='h')
    sector_sentiment_trend = sector_sentiment_trend.reindex(all_hours)

    # Apply centered rolling average to smooth the "all over the place" lines
    smoothed_trends = sector_sentiment_trend.rolling(window=rolling_window, min_periods=4, center=True).mean()

    # 5. VISUALIZATION
    _, ax = plt.subplots(figsize=(14, 7), facecolor='#1e1e1e')
    ax.set_facecolor('#1e1e1e')

    for i, sector in enumerate(smoothed_trends.columns):
        ax.plot(smoothed_trends.index, smoothed_trends[sector],
                label=sector, linewidth=2.5,
                color=muted_colors[i % len(muted_colors)], alpha=0.9)

    # Formatting
    ax.set_title(f'Sector Sentiment Trend Identification ({rolling_window}h Smoothing)',
                 fontsize=16, color='#d1d1d1', pad=20)
    ax.set_ylabel('Avg Sentiment Score', color='#d1d1d1')
    ax.axhline(0, color='white', linestyle='--', alpha=0.2)

    # Date formatting for the X-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    plt.xticks(color='#d1d1d1', rotation=45)
    plt.yticks(color='#d1d1d1')

    # Grid and Spines
    ax.grid(color='white', linestyle=':', alpha=0.1)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(title='Market Sectors', loc='upper left', bbox_to_anchor=(1, 1), frameon=False)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()


def merge_df(wdf, sdf, resample):
    """Combines comment and market dataframes"""
    wdf['datetime'] = pd.to_datetime(wdf['datetime'])
    sdf['timestamp'] = pd.to_datetime(sdf['timestamp'])

    wdf.set_index('datetime', inplace=True)
    sdf.set_index('timestamp', inplace=True)

    # Resample SPY DF
    ohlc_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    spy = sdf.resample(f'{resample}T').agg(ohlc_dict)
    spy.dropna(subset=['close'], inplace=True)

    # Resample WSB DF
    sentiment_dict = {
        'ensemble_sentiment': 'mean',
        'nltk_sentiment': 'mean',
        'score_finbert': 'mean',
        'score_vader': 'mean',
        'score': 'sum',      # Total upvotes/engagement in this window
        'text': 'count'      # Count the number of posts (volume of chatter)
    }
    wsb = wdf.resample(f'{resample}T').agg(sentiment_dict)
    wsb.index = wsb.index.tz_localize('EST')
    wsb.rename(columns={'text': 'post_volume'}, inplace=True)

    merged_df = spy.join(wsb, how='left')
    merged_df['nltk_sentiment'] = merged_df['nltk_sentiment'].fillna(0)
    merged_df['ensemble_sentiment'] = merged_df['ensemble_sentiment'].fillna(0)
    merged_df['score_finbert'] = merged_df['score_finbert'].fillna(0)
    merged_df['score_vader'] = merged_df['score_vader'].fillna(0)
    merged_df['post_volume'] = merged_df['post_volume'].fillna(0)
    merged_df['score'] = merged_df['score'].fillna(0)

    return merged_df


def normalize_sentiment(df, sentiment_cols):

    for col in sentiment_cols:
        col_min = df[col].min()
        col_max = df[col].max()

        if col_max != col_min:
            df[f'{col}_norm'] = 2 * ((df[col] - col_min) / (col_max - col_min)) - 1
        else:
            df[f'{col}_norm'] = 0.0

    return df


def plot_sentiment_vs_price(df: pd.DataFrame,
                            title: str = "Price vs. Finbert & Ensemble Sentiment",
                            save_path: str = None):
    """Plots Price as a line and Sentiment as a scatter plot to eliminate
    misleading lines across time gaps.

    Parameters
    ----------
    df        : merged DataFrame with price and sentiment columns
    title     : chart title
    save_path : optional file path; if supplied the figure is saved before
                being displayed (e.g. 'results/price_vs_sentiment.png').
    """
    # 1. Targeted Column Selection
    price_col = 'close' if 'close' in df.columns else ('open' if 'open' in df.columns else None)
    target_sentiments = ['score_finbert', 'ensemble_sentiment']
    sent_cols = [c for c in df.columns if any(target in c.lower() for target in target_sentiments) 
                 and '_norm' not in c.lower()]

    if not price_col:
        print("Required price data not found.")
        return

    # 2. Visual Styling
    bg_color = "#121212"
    text_color = "#B0B0B0"
    grid_color = "#2A2A2A"
    price_color = "#6D8299"
    sent_colors = ["#E28E8E", "#8E9775"]

    # Use positional indexing (0, 1, 2...) to collapse time gaps (weekends/nights)
    x_axis = np.arange(len(df))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 8), sharex=True, facecolor=bg_color)
    fig.subplots_adjust(hspace=0.15)

    # --- TOP PANEL: CLOSE PRICE ---
    ax1.set_facecolor(bg_color)
    ax1.plot(x_axis, df[price_col], color=price_color, linewidth=1.5, label="Price (Close)")
    ax1.set_ylabel("Price ($)", color=text_color, fontweight='bold')
    ax1.set_title(title, color=text_color, fontsize=16, pad=20)

    # --- BOTTOM PANEL: TARGETED SENTIMENTS (SCATTER) ---
    ax2.set_facecolor(bg_color)
    for i, col in enumerate(sent_cols):
        # We use scatter() here to represent individual sentiment events
        ax2.scatter(x_axis, df[col], 
                    label=col.replace('_', ' ').title(),
                    color=sent_colors[i % len(sent_colors)], 
                    s=25,          # Point size
                    alpha=0.7,     # Transparency to see overlapping points
                    edgecolors='none')

    # Static zero line for sentiment reference
    ax2.axhline(0, color=text_color, linewidth=0.8, linestyle='--', alpha=0.3)
    ax2.set_ylabel("Sentiment Score", color=text_color, fontweight='bold')
    ax2.set_xlabel("Time Index", color=text_color)

    # --- FIX X-AXIS LABELS ---
    # Map the integer index back to readable timestamps
    num_ticks = 10
    tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
    tick_labels = df.index[tick_indices].strftime('%Y-%m-%d %H:%M')
    ax2.set_xticks(tick_indices)
    ax2.set_xticklabels(tick_labels, rotation=15, ha='right')

    # 3. UI Refinement
    for ax in [ax1, ax2]:
        ax.tick_params(axis='both', colors=text_color, labelsize=10)
        ax.grid(True, linestyle=':', alpha=0.15, color=text_color)
        for spine in ax.spines.values():
            spine.set_edgecolor(grid_color)
        
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1), facecolor='#1A1A1A',
                  edgecolor=grid_color, labelcolor=text_color, fontsize=11)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.show()