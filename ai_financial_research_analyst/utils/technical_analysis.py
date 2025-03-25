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
    df = dropna(df)

    if len(df) == 0:
        st.write("Warning: No data returned from yfinance!")
    
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

def add_indicators(df: pd.DataFrame, sma_period: int) -> pd.DataFrame:
    """Add technical indicators to the dataframe."""
    # Calculate SMA
    df[f'SMA_{sma_period}'] = df['Close'].rolling(window=sma_period).mean()
    return df

def create_technical_analysis_chart(df: pd.DataFrame, patterns: dict, sma_period: int) -> go.Figure:
    """Creates a combined technical analysis chart with all detected patterns and indicators."""
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

    # Add SMA
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df[f'SMA_{sma_period}'],
            mode='lines',
            name=f'SMA ({sma_period})',
            line=dict(color='orange', width=2)
        )
    )

    # Add double top pattern if detected
    if patterns['double_top']['detected']:
        top1, top2 = patterns['double_top']['points']
        neckline = patterns['double_top']['neckline']
        
        # Add peaks
        fig.add_trace(
            go.Scatter(
                x=[df.index[top1], df.index[top2]],
                y=[df['Close'].values[top1], df['Close'].values[top2]],
                mode='markers',
                name='Double Top',
                marker=dict(color='black', size=12, symbol='x')
            )
        )
        
        # Add extended neckline
        fig.add_trace(
            go.Scatter(
                x=[df.index[top1], df.index[-1]],
                y=[neckline, neckline],
                mode='lines',
                name='Double Top Neckline',
                line=dict(color='black', dash='dash')
            )
        )

    # Add double bottom pattern if detected
    if patterns['double_bottom']['detected']:
        bot1, bot2 = patterns['double_bottom']['points']
        neckline = patterns['double_bottom']['neckline']
        
        # Add troughs
        fig.add_trace(
            go.Scatter(
                x=[df.index[bot1], df.index[bot2]],
                y=[df['Close'].values[bot1], df['Close'].values[bot2]],
                mode='markers',
                name='Double Bottom',
                marker=dict(color='Black', size=12, symbol='x')
            )
        )
        
        # Add extended neckline
        fig.add_trace(
            go.Scatter(
                x=[df.index[bot1], df.index[-1]],
                y=[neckline, neckline],
                mode='lines',
                name='Double Bottom Neckline',
                line=dict(color='black', dash='dash')
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Technical Analysis Chart",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        showlegend=True,
        height=600,
        hovermode='x unified',
        xaxis=dict(
            rangeslider=dict(
                visible=False
            )
        )
    )

    return fig

def analyze_stock(ticker: str, period: str, interval: str, sma_period: int) -> dict:
    """Fetch stock data and analyze support, resistance, chart patterns, and indicators."""
    # Fetch complete OHLC data
    df = fetch_stock_data(ticker, period, interval)
    
    # Add technical indicators
    df = add_indicators(df, sma_period)
    
    # Get support and resistance levels using Close prices
    support, resistance = identify_support_resistance(df)
    
    # Detect patterns using Close prices
    is_double_top, top1, top2, top_neckline = detect_double_top(df)
    is_double_bottom, bot1, bot2, bottom_neckline = detect_double_bottom(df)
    
    # Create patterns dictionary
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
    fig = create_technical_analysis_chart(df, patterns, sma_period)
    
    # Save the chart as a temporary PNG file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
        fig.write_image(tmp_file.name)
        chart_path = tmp_file.name
    
    # Display the chart
    st.plotly_chart(fig, use_container_width=True)
    
    return {
        "support_levels": support.tolist(),
        "resistance_levels": resistance.tolist(),
        "double_top": is_double_top,
        "double_bottom": is_double_bottom,
        "chart_path": chart_path
    }
