import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from ta import add_all_ta_features
from ta.utils import dropna
from scipy.signal import find_peaks
import streamlit as st 

def fetch_stock_data(ticker: str, period: str, interval: str ) -> pd.DataFrame:
    """Fetch historical stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    raw_data = stock.history(period=period, interval=interval)

    df = raw_data[['Open', 'High', 'Low', 'Close']] 
    
    return df

def identify_support_resistance(df: pd.DataFrame) -> tuple:
    """Identify potential support and resistance levels using peaks and troughs."""
    prices = df['Close'].values
    peaks, _ = find_peaks(prices, distance=10)
    troughs, _ = find_peaks(-prices, distance=10)
    
    resistance_levels = prices[peaks][-3:]
    support_levels = prices[troughs][-3:]
    
    return support_levels, resistance_levels

def detect_double_top(df: pd.DataFrame, distance: int = 4, height_tolerance: float = 0.02) -> tuple:
    """Detects double top pattern in the given price data."""
    prices = df['Close'].values
    peaks, _ = find_peaks(prices, distance=distance)
    
    if len(peaks) < 2:
        return False, None, None, None

    peak_prices = prices[peaks]
    sorted_indices = np.argsort(peak_prices)[-2:]
    top1, top2 = sorted(peaks[sorted_indices])

    if abs(prices[top1] - prices[top2]) > height_tolerance * prices[top1]:
        return False, None, None, None

    neckline = min(prices[top1:top2])
    post_peak_prices = prices[top2:]

    if not any(post_peak_prices < neckline):
        return False, None, None, None

    return True, top1, top2, neckline

def detect_double_bottom(df: pd.DataFrame, distance: int = 4, height_tolerance: float = 0.02) -> tuple:
    """Detects double bottom pattern in the given price data."""
    prices = df['Close'].values
    troughs, _ = find_peaks(-prices, distance=distance)

    if len(troughs) < 2:
        return False, None, None, None

    trough_prices = prices[troughs]
    sorted_indices = np.argsort(trough_prices)[:2]
    bot1, bot2 = sorted(troughs[sorted_indices])

    if abs(prices[bot1] - prices[bot2]) > height_tolerance * prices[bot1]:
        return False, None, None, None

    neckline = max(prices[bot1:bot2])
    post_trough_prices = prices[bot2:]

    if not any(post_trough_prices > neckline):
        return False, None, None, None

    return True, bot1, bot2, neckline

def add_indicators(df: pd.DataFrame, selected_indicators: list) -> pd.DataFrame:
    """Add selected technical indicators to the dataframe."""
    # Simple Moving Averages
    sma_periods = {
        "SMA10": 10,
        "SMA20": 20,
        "SMA50": 50,
        "SMA100": 100,
        "SMA200": 200
    }
    
    # Calculate selected SMAs
    for indicator in selected_indicators:
        if indicator in sma_periods:
            period = sma_periods[indicator]
            df[indicator] = df['Close'].rolling(window=period).mean()

    
    # Bollinger Bands
    if "Bollinger Bands" in selected_indicators:
        df['BB_middle'] = df['Close'].rolling(window=20).mean()
        df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
        df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()
    
    return df

def create_technical_analysis_chart(df: pd.DataFrame, patterns: dict, selected_indicators: list) -> go.Figure:
    """Creates a combined technical analysis chart with selected indicators."""
    fig = go.Figure()

    # Add candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='OHLC'
        )
    )

    # Add double top pattern if detected
    if patterns['double_top']['detected']:
        top1, top2 = patterns['double_top']['points']
        neckline = patterns['double_top']['neckline']
        
        # Add markers for tops
        fig.add_trace(go.Scatter(
            x=[df.index[top1], df.index[top2]],
            y=[df['Close'].iloc[top1], df['Close'].iloc[top2]],
            mode='markers',
            marker=dict(symbol="x", size=10, color='black'),
            name='Double Top'
        ))
        
        # Add neckline
        fig.add_trace(go.Scatter(
            x=[df.index[top1], df.index[top2]],
            y=[neckline, neckline],
            mode='lines',
            line=dict(dash='dash', color='black'),
            name='Double Top Neckline'
        ))

    # Add double bottom pattern if detected
    if patterns['double_bottom']['detected']:
        bot1, bot2 = patterns['double_bottom']['points']
        neckline = patterns['double_bottom']['neckline']
        
        # Add markers for bottoms
        fig.add_trace(go.Scatter(
            x=[df.index[bot1], df.index[bot2]],
            y=[df['Close'].iloc[bot1], df['Close'].iloc[bot2]],
            mode='markers',
            marker=dict(symbol="x", size=10, color='blue'),
            name='Double Bottom'
        ))
        
        # Add neckline
        fig.add_trace(go.Scatter(
            x=[df.index[bot1], df.index[bot2]],
            y=[neckline, neckline],
            mode='lines',
            line=dict(dash='dash', color='blue'),
            name='Double Bottom Neckline'
        ))

    # Add selected indicators
    colors = {
        "SMA10": "#1f77b4",
        "SMA20": "#ff7f0e",
        "SMA50": "#2ca02c", 
        "SMA100": "#9467bd",
        "SMA200": "#d62728"
    }

    # Add Moving Averages
    for indicator in selected_indicators:
        if indicator.startswith(("SMA")):
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[indicator],
                    mode='lines',
                    name=indicator,
                    line=dict(color=colors.get(indicator, '#000000'))
                )
            )

    # Add Bollinger Bands
    if "Bollinger Bands" in selected_indicators:
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], name='BB Upper',
                               line=dict(color='gray', dash='dash')))
        fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], name='BB Lower',
                               line=dict(color='gray', dash='dash')))


    # Update layout based on selected indicators
    layout_updates = {
        "title": "Technical Analysis Chart",
        "xaxis_title": "Date",
        "yaxis_title": "Price (USD)",
        "showlegend": True,
        "hovermode": 'x unified',
        "xaxis_rangeslider_visible": False  # Disable the range slider
    }

    fig.update_layout(**layout_updates)

    return fig

def analyze_stock(ticker: str, period: str, interval: str, selected_indicators: list) -> dict:
    """Fetch stock data and analyze with selected indicators."""
    df = fetch_stock_data(ticker, period, interval)
    
    if df.empty:
        st.error(f"No data available for {ticker}. Please select another data interval or ticker.")
        return None
    
    # Add technical indicators
    df = add_indicators(df, selected_indicators)
    
    # Get support and resistance levels
    support, resistance = identify_support_resistance(df)
    
    # Detect patterns
    is_double_top, top1, top2, top_neckline = detect_double_top(df)
    is_double_bottom, bot1, bot2, bottom_neckline = detect_double_bottom(df)
    
    patterns = {
        'double_top': {
            'detected': is_double_top,
            'points': (top1, top2) if is_double_top else None,
            'neckline': top_neckline
        },
        'double_bottom': {
            'detected': is_double_bottom,
            'points': (bot1, bot2) if is_double_bottom else None,
            'neckline': bottom_neckline
        }
    }
    
    # Create and display the chart
    fig = create_technical_analysis_chart(df, patterns, selected_indicators)
    
    return {
        "support_levels": support.tolist() if len(support) > 0 else [],
        "resistance_levels": resistance.tolist() if len(resistance) > 0 else [],
        "double_top": is_double_top,
        "double_bottom": is_double_bottom,
        "figure": fig
    }
