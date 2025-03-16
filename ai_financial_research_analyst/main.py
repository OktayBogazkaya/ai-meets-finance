import time
import os
import json
import tempfile

import pandas as pd
import requests
import streamlit as st
import yt_dlp

from google import genai
from google.genai.types import GenerateContentConfig
from google.genai import types

from pydantic import BaseModel
from typing import Optional

from google.genai import types


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
    
def save_uploaded_audio(uploaded_file):
    """Save uploaded file to a temporary file and return the path."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
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
    st.sidebar.warning("🚨 GEMINI API Key is missing! Please provide a valid Gemini API key to proceed.")

# Tabs for different sections
tabs = ["📞 Earnings Call Analysis", "🎙️ Podcast Analysis", "🎥 Youtube Video Analysis"]
tab1, tab2, tab3 = st.tabs(tabs)

# Display content based on selected tab
with tab1:
    # Get Earnings Data
    st.subheader("Analyze Earnings Calls")

    col1, col2, col3 = st.columns(3)

    with col1:
        symbol = st.selectbox("Enter Stock Symbol", ["PLTR", "TSLA", "NVDA", "MSFT", "AAPL", "META"])
    with col2:
        year = st.selectbox("Select Year", [2024, 2023, 2022])
    with col3:
        quarter = st.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    if st.button("Process Earnings Call"):
        if not FMP_API_KEY:
            st.error("🚨 FMP API Key is missing! Please provide a valid API key to proceed.")
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
    st.subheader("Analyze Podcasts")
    audio_file = st.file_uploader("Upload Podcast File (MP3)", type=['mp3'])
    if audio_file is not None:
        audio_path = save_uploaded_audio(audio_file)
        st.subheader("Play Podcast Episode")
        st.audio(audio_path)

        if st.button("Process Podcast"):
            if not GEMINI_API_KEY:
                st.error("🚨 Gemini API Key is missing! Please provide a valid API key to proceed.")
            else:
                with st.spinner('Processing...'):
                    audio_file = client.files.upload(file=audio_path)

                    # Define system instruction
                    system_instruction = """
                    You are a stock market analyst who analyzes financial related podcasts to find investment ideas. Provides a comprehensive summary of the topics covered in teh podcast.
                
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
                    You are a stock market analyst who analyzes market sentiment given the earnings call transcripts. Based on the transcripts, extract all mentioned companies
                    
                    - For each company you extracted, provide the name, indicate if it's publicly traded, and if applicable, provide the stock symbol
                    - Give a score 1 if the sentiment is positive, -1 if neagtive, and 0 if the sentiment is neutral towards the mentioned company
                    - Reiterate the statement
                    - Gove a one sentence explanation

                    Exclude company names which are only mentioned during analyst introductions.
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

with tab3:
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
                st.error("🚨 FMP API Key is missing! Please provide a valid API key to proceed.")
            else:
                with st.spinner('Processing...'):
                    
                    # Define system instruction
                    prompt = """
                        You are a financial market analyst. Analyze this video for financial insights and provide:
                    
                        1. Summary of the main topics
                        2. Key insights shared
                        3. Any financial advice or strategies mentioned

                        Your reporting style:
                        - Highlight key insights with bullet points
                        - Include technical term explanations
                        - Commentary and insights should be succinct but informative
                        """

                    response = client.models.generate_content(
                        model=model, 
                        contents=types.Content(
                            parts=[
                                types.Part(text=prompt),
                                types.Part(
                                    file_data=types.FileData(file_uri=video_url)
                                )
                            ]
                        )
                    )
                    
                    st.subheader(f"Video Summary:")
                    st.write(response.text)

                    #Show token usage 
                    with st.expander("🔍 Show Token Usage", expanded=False):
                        # Output Token Count
                        input_token_count(response)