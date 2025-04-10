# 🕵🏻‍♂️ AI Financial Research Analyst

A Streamlit app that leverages Google's multimodal Flash models to help investors process and analyze financial information in multiple formats, extracting key insights with ease.

## Web App

Try it yourself 👉 [Streamlit App](https://ai-financial-research-analyst.streamlit.app/)

## Features

- **Earnings Call Transcript Analysis** - Extract insights from transcripts using the FinancialModelingPrep (FMP) API
- **Image&Technical Chart Analysis** - Extract insights based on images or perform technical chart analysis to detect patterns and get AI-powered summary with trading recommendations
- **Podcast Analysis** – Upload audio files and get AI-powered summaries
- **Video Analysis** – Process YouTube video URLs to analyze financial discussions, interviews, and more
- **Token Count Tracking** – Get input, output and total tokens for each LLM interaction

## 🛫 Getting Started

1. Clone the repository

```
git clone https://github.com/OktayBogazkaya/ai-meets-finance.git

```

2. Navigate to the desired project directory

```
cd ai-meets-finance/ai_financial_research_analyst

```

3. Install dependencies

```
pip install -r requirements.txt
```

4. Get your Gemini API Key

- Generate an API key from [Google AI Studio](https://ai.google.dev/gemini-api/docs/api-key)

5. Get your FMP API Key (to use the Earnings Call Transcript feature)

- Sign up and generate an API key on [FMP](https://site.financialmodelingprep.com/)
