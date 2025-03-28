import time
import os
import json
import tempfile

import pandas as pd
import requests
import streamlit as st
from PIL import Image

from google import genai
from google.genai.types import GenerateContentConfig, Part
from google.genai import types

from pydantic import BaseModel
from typing import Optional

from google.genai import types

from utils.technical_analysis import analyze_stock


# Streamlit page instructions
st.set_page_config(
    page_title="Financial Analyst",
    page_icon="🕵🏻‍♂️",
    layout="wide"
)

# Company class model 
class Company(BaseModel):
    """
    A model representing a company with relevant attributes
    """

    name: str
    symbol: Optional[str]
    public: bool
    sector: Optional[str]
    industry: Optional[str]
    sentiment: int
    note: str

# Function to get earnings call transcripts
def get_earnings_calls(symbol, year, quarter, api_key):
    endpoint = f"https://financialmodelingprep.com/api/v3/earning_call_transcript/{symbol}?year={year}&quarter={quarter}&apikey={api_key}"
    response = requests.get(endpoint)
    if response.status_code == 200:
        return response.json()
    else:
        st.error("Error fetching earnings call data")
        return None
    
def save_uploaded_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

def save_image_file(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            # Save the Plotly figure as PNG
            uploaded_file.write_image(tmp_file.name)
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Input Token Count
def input_token_count(response):
    token_count = response.usage_metadata

    input_tokens = token_count.prompt_token_count
    output_tokens = token_count.candidates_token_count
    total_token_count = token_count.total_token_count

    st.write(f"🔸 Input Tokens: {input_tokens}")
    st.write(f"🔸 Output Tokens: {output_tokens}")
    st.write(f"🔸 Total Tokens: {total_token_count}")


# Streamlit app title
st.title("🕵🏻‍♂️ AI-Powered Financial Research Analyst")

# Sidebar for API keys
st.sidebar.header("API Keys")
FMP_API_KEY = st.sidebar.text_input("Enter your FMP API Key", type="password")
st.sidebar.info("Get your FMP API key [here](https://financialmodelingprep.com)")
GEMINI_API_KEY = st.sidebar.text_input("Enter your Gemini API Key", type="password")
st.sidebar.info("Get your Gemini API key [here](https://ai.google.dev/gemini-api/docs/api-key)")

if GEMINI_API_KEY:
    client = genai.Client(api_key=GEMINI_API_KEY)
    model = "gemini-2.0-flash"
    config = {"response_modalities": ["TEXT"]}
else:
    st.sidebar.warning("⚠️ Please provide a valid Gemini API key to proceed.")

# Tabs for different sections
tabs = ["📞 Earnings Call Analysis", "📸 Image/Chart Analysis", "🎙️ Podcast Analysis", "🎥 Youtube Video Analysis"]
tab1, tab2, tab3, tab4 = st.tabs(tabs)

# Display content based on selected tab
with tab1:
    # Get Earnings Data
    st.subheader("Analyze Earnings Calls")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox("Select Ticker", ["ASML","FIVE", "DVN", "PLTR", "TSLA", "NVDA", "MSFT", "AAPL", "META"])
    with col2:
        year = st.selectbox("Select Year", [2024, 2023, 2022])
    with col3:
        quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    if st.button("Process Earnings Call"):
        if not FMP_API_KEY:
            st.error("⚠️ FMP API Key is missing! Please provide a valid API key in the side bar to proceed.")
        else:
            with st.spinner('Processing...'):
                transcript = get_earnings_calls(symbol, year, quarter, FMP_API_KEY)
                if not transcript:
                    st.warning("⚠️ No earnings call transcript found for this company and quarter.")
                else:
                    # Define system instruction
                    system_instruction = """
                    You are a stock market analyst who analyzes earnings call transcripts and provides a comprehensive summary.

                    Follow these steps for comprehensive financial analysis:

                    1. Financial Performance Summary
                    2. Product and Service Breakdown 
                    3. Geographical Breakdown
                    4. Challenges and Risks
                    5. Future Outlook
                    6. Other Topics
                
                    Your reporting style:
                    - Highlight key insights with bullet points
                    - Use tables for data presentation
                    - Include technical term explanations
                    - Commentary and insights should be succinct but informative
                    """

                    response = client.models.generate_content(
                        model=model, 
                        contents=transcript[0]['content'], 
                        config=GenerateContentConfig(
                            system_instruction = system_instruction
                        ),
                    )
                    st.subheader(f"{quarter} {year} Earnings Call Summary:")
                    st.write(response.text)

                    # Show token usage
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        # Output Token Count
                        input_token_count(response)

                    # Define system instruction
                    system_instruction = """
                    You are a stock market analyst who analyzes market sentiment given the earnings call transcripts.
                    
                    - Extract all of the companies mentioned, including the company name, the ticker symbol, whether they are publicly traded or nor not, the industry and sector they are operating in.
                    - Analyze the sentiment for each company extracted, and give a score 1 if the sentiment is positive, -1 if neagtive, and 0 if the sentiment is neutral. Reiterate the statement
                    - Give a one sentence explanation for teh senntiment score

                    Exclude company names which are only mentioned during analyst introductions.
                    """

                    response = client.models.generate_content(
                        model=model, 
                        contents=transcript[0]['content'], 
                        config=GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type= "application/json",
                            response_schema= list[Company]
                            )
                    )    

                    st.subheader("Companies Mentioned:")

                    json_output = json.loads(response.text)
                    df_data = pd.DataFrame(json_output)
                    st.write(df_data)

                    # Show token usage
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        # Output Token Count
                        input_token_count(response)


with tab2:
    st.subheader("Analyze Images&Charts")
    analysis_type = st.radio(
        
        "Please select",
        ["Image Analysis", "Technical Chart Analysis"],
        captions=[
            "Get a summary of the key insights",
            "Get a comprehensive technical analysis of the stock chart"
        ]                            
    )

    if analysis_type == "Image Analysis":
        image_file = st.file_uploader("Upload Image")
        if image_file is not None:
            image_path = save_uploaded_file(image_file)
            st.subheader("Image Preview")
            img = Image.open(image_path)
            st.image(img, "Uploaded image")

            if st.button("Process Image"):
                if not GEMINI_API_KEY:
                    st.error("⚠️ Gemini API Key is missing! Please provide a valid API key in the side bar to proceed.")
                else:
                    with st.spinner('Processing...'):
                        image_path = client.files.upload(file=image_path)

                        # Define system instruction
                        system_instruction = """
                        You are a financial market analyst. Analyze this image for financial insights and provide:
                
                        1. Executive Summary
                        {Concise overview of key findings and significance}

                        2. Key Findings
                        {Main discoveries and analysis}
                        {Expert insights and quotes}

                        Your reporting style:
                        - Highlight key insights with bullet points
                        - Include technical term explanations
                        """

                        response = client.models.generate_content(
                            model=model, 
                            contents=image_path, 
                            config=GenerateContentConfig(
                                system_instruction = system_instruction
                            ),
                        )
                        st.subheader(f"Image Analyis:")
                        st.write(response.text)

                        # Show token usage
                        with st.expander("🔍 Show Token Usage", expanded=False):
                            # Output Token Count
                            input_token_count(response)

    if analysis_type == "Technical Chart Analysis":
        st.subheader("Basic Settings")

        col1, col2, col3 = st.columns(3)
        with col1: 
            symbol = st.selectbox("Select Ticker", ["ASML","FIVE","DVN","PLTR", "TSLA", "NVDA", "MSFT", "ISRG", "AAPL", "META", "ETH-USD"])
            
            indicators = st.multiselect(
                "Select Technical Indicators",
                options=[
                    "SMA10", "SMA20", "SMA50", "SMA100", "SMA200",
                    "Bollinger Bands"
                ],
                default=["SMA20"],
                help="Choose multiple technical indicators to display on the chart"
            )

        with col2:
            period = st.selectbox("Select Data Range", ["1d","5d","1mo", "3mo", "6mo", "1y", "2y","5y", "ytd", "max"])
        with col3:
            interval = st.selectbox("Select Data Interval", 
                                    ["1h","1d","1wk","1mo", "3mo"]
            )

        # Get technical analysis results
        result = analyze_stock(symbol, period, interval, indicators)
        
        if result is not None:
            st.plotly_chart(result['figure'], use_container_width=True)
            
        if symbol is not None:
            if st.button("Start Analysis"):
                if not GEMINI_API_KEY:
                    st.error("⚠️ Gemini API Key is missing! Please provide a valid API key in the side bar to proceed.")
                else:
                    image_path = save_image_file(result['figure'])

                    # Read content from temp_file
                    with open(image_path, 'rb') as f:
                        local_file_img_bytes = f.read()

                    with st.spinner('Processing...'):
                        # Get technical analysis results and chart path

                            # Define system instruction for technical analysis
                            system_instruction = """
                            You are an expert technical analyst. Analyze this stock chart and provide:

                            1. Technical Analysis Summary
                            A detailed explanation of your analysis, including: 
                            - Trend analysis
                            - Pattern identification

                            2. Trading Recommendation
                            Based on your analysis and the the chart, provide:
                            - BUY, SELL, or HOLD recommendation
                            - Detailed rationale for the recommendation
                            - Key risk factors to consider
                            - Suggested entry/exit points if the recommendation is BUY or SELL. Otherwise skip this section.

                            Format your response in a clear, structured manner with bullet points.
                            Be specific about price levels and technical indicators.
                            """

                            response = client.models.generate_content(
                                model=model, 
                                contents=[Part.from_bytes(data=local_file_img_bytes, mime_type="image/png")], 
                                config=GenerateContentConfig(
                                    system_instruction = system_instruction
                                ),
                            )

                            # Display the analysis
                            st.subheader("Technical Analysis Summary")
                            st.write(response.text)

                            # Show token usage
                            with st.expander("🔍 Show Token Usage", expanded=False):
                                input_token_count(response)

                            # Clean up the temporary file
                            os.unlink(image_path)
                        
with tab3:
    st.subheader("Analyze Podcasts")
    audio_file = st.file_uploader("Upload Podcast File (MP3)", type=['mp3'])
    if audio_file is not None:
        audio_path = save_uploaded_file(audio_file)
        st.subheader("Play Podcast Episode")
        st.audio(audio_path)

        if st.button("Process Podcast"):
            if not GEMINI_API_KEY:
                st.error("⚠️ Gemini API Key is missing! Please provide a valid API key in the side bar to proceed.")
            else:
                with st.spinner('Processing...'):
                    audio_file = client.files.upload(file=audio_path)

                    # Define system instruction
                    system_instruction = """
                    You are a financial market analyst. Analyze this podcast for financial insights and provide:
            
                    1. Executive Summary
                    {Concise overview of key findings and significance}

                    2. Key Findings
                    {Main discoveries and analysis}
                    {Expert insights and quotes}

                    Your reporting style:
                    - Highlight key insights with bullet points
                    - Include technical term explanations
                    - Commentary and insights should be succinct but informative
                    """

                    response = client.models.generate_content(
                        model=model, 
                        contents=audio_file, 
                        config=GenerateContentConfig(
                            system_instruction = system_instruction
                        ),
                    )
                    st.subheader(f"Podcast Summary:")
                    st.write(response.text)

                    # Show token usage
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        # Output Token Count
                        input_token_count(response)

                    # Define system instruction
                    system_instruction = """
                    You are a stock market analyst who analyzes market sentiment given the podcast interview. Based on the interview, extract all mentioned companies.
                    
                    - For each company you extracted, provide the name, indicate if it's publicly traded, and if applicable, provide the stock symbol
                    - Give a score 1 if the sentiment is positive, -1 if neagtive, and 0 if the sentiment is neutral towards the mentioned company
                    - Reiterate the statement
                    - Give a one sentence explanation

                    """
                    response = client.models.generate_content(
                        model=model, 
                        contents=audio_file,
                        config=GenerateContentConfig(
                            system_instruction=system_instruction,
                            response_mime_type= "application/json",
                            response_schema= list[Company]
                            )
                    )

                    st.subheader("Companies Mentioned:")

                    json_output = json.loads(response.text)
                    df_data = pd.DataFrame(json_output)
                    st.write(df_data)

                    # Show token usage
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        input_token_count(response)

with tab4:
    st.subheader("Analyze YT Videos")

    # Input YouTube link
    video_url = st.text_input("Enter YouTube Video URL:")

    if video_url:
        try:
            # Display video
            st.video(video_url)

        except Exception as e:
            st.error(f"An error occurred: {e}")

        if st.button("Process Video"):
            if not GEMINI_API_KEY:
                st.error("⚠️ Gemini API Key is missing! Please provide a valid API key in the side bar to proceed.")
            else:
                with st.spinner('Processing...'):
                    
                    # Define system instruction
                    system_instruction = """
                        You are a financial market analyst. Analyze this video for financial insights and provide:
                    
                        1. Executive Summary
                        {Concise overview of key findings and significance}

                        2. Key Findings
                        {Main discoveries and analysis}
                        {Expert insights and quotes}

                        Your reporting style:
                        - Highlight key insights with bullet points
                        - Include technical term explanations
                        - Commentary and insights should be succinct but informative
                        """

                    response = client.models.generate_content(
                        model=model, 
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    file_data=types.FileData(file_uri=video_url)
                                )
                            ]
                        ),
                        config=GenerateContentConfig(
                            system_instruction = system_instruction,
                        )
                    )
                    
                    st.subheader(f"Video Summary:")
                    st.write(response.text)

                    #Show token usage 
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        # Output Token Count
                        input_token_count(response)

                    # Define system instruction
                    system_instruction = """
                    You are a stock market analyst who analyzes market sentiment given the earnings call transcripts. Based on the transcripts, extract all mentioned companies.
                    
                    - For each company you extracted, provide the name, indicate if it's publicly traded, and if applicable, provide the stock symbol
                    - Give a score 1 if the sentiment is positive, -1 if neagtive, and 0 if the sentiment is neutral towards the mentioned company
                    - Reiterate the statement
                    - Gove a one sentence explanation

                    Exclude company names which are only mentioned during analyst introductions.
                    """
                    response = client.models.generate_content(
                        model=model, 
                        contents=types.Content(
                            parts=[
                                types.Part(
                                    file_data=types.FileData(file_uri=video_url)
                                )
                            ]
                        ),
                        config=GenerateContentConfig(
                            system_instruction = system_instruction,
                            response_mime_type= "application/json",
                            response_schema= list[Company]
                            )
                    )

                    st.subheader("Companies Mentioned:")

                    json_output = json.loads(response.text)
                    df_data = pd.DataFrame(json_output)
                    st.write(df_data)

                    # Show token usage
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        input_token_count(response)
