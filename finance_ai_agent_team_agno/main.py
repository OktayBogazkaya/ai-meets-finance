import streamlit as st
from textwrap import dedent

# Import Agno AI components
from agno.agent import Agent
from agno.team import Team
from agno.models.google import Gemini
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.yfinance import YFinanceTools
from agno.tools.newspaper4k import Newspaper4kTools

# Set page config
st.set_page_config(
    page_title="Financial Analysis Team",
    page_icon="📊",
    layout="wide"
)

# Sidebar for API keys
st.sidebar.header("API Configuration")

# Check for API key in secrets first
try:
    gemini_api_key = st.secrets["GEMINI_API_KEY"]
    st.sidebar.success("API KEY detected in secrets!")
except KeyError:
    gemini_api_key = st.sidebar.text_input(
        "Enter API Key (or add to .streamlit/secrets.toml)",
        type="password"
    )

# Agent definitions
st.title("AI Financial Analysis Team")
st.markdown("Get comprehensive market analysis combining stock data and news insights.")

# News Sentiment Agent
news_sentiment_agent = Agent(
    name="News Sentiment Agent",
    model=Gemini(
        id="gemini-2.0-flash",
        api_key=gemini_api_key
    ),
    description="You are a News Sentiment Decoding Assistant.",
    tools=[Newspaper4kTools()],
    instructions=["Decode the news and provide the sentiment ranging from +10 to -10 in table format with the following columns Date, Time, News, Source and Score. Also provide reasoning explanation point by point after the Table."],
    add_datetime_to_instructions=True,
    add_history_to_messages=True,
    num_history_responses=5,
    markdown=True,
    exponential_backoff=True
)

web_agent = Agent(
        name="Web Agent",
        description="You are an web search expert at finding information online.",
        model=Gemini(
            id="gemini-2.0-flash",
            api_key=gemini_api_key
        ),
            tools=[DuckDuckGoTools(search=True, news=False)],
            instructions=["Always include sources"],
            add_datetime_to_instructions=True,
            add_history_to_messages=True,
            num_history_responses=5,
            markdown=True,
            exponential_backoff=True
        )

finance_agent = Agent(
        name="Finance Agent",
        description="You are an investment analyst that researches stock prices, analyst recommendations, and stock fundamentals.",
        model=Gemini(
            id="gemini-2.0-flash",
            api_key=gemini_api_key
        ),
        tools=[
            YFinanceTools(
                stock_price=True,
                analyst_recommendations=True,
                stock_fundamentals=True,
                historical_prices=False,
                company_info=True,
                company_news=True,
            )
        ],
        instructions=["Format your response using markdown and use tables to display data where possible."],
        add_datetime_to_instructions=True,
        add_history_to_messages=True,
        num_history_responses=5,
        markdown=True,
        exponential_backoff=True
    )

def initialize_team():
        """Initialize the team of agents"""
        return Team(
            name="Financial Analysis Team",
            mode="coordinate",
            model=Gemini(
                id="gemini-2.0-flash",
                api_key=gemini_api_key
            ),
            members=[web_agent,
                    finance_agent,
                    news_sentiment_agent
                    ],
            description="You are a task router that coordinates specialized agents to provide a comprehensive financial analysis",
            instructions=dedent("""\
                
                You are a Financial News and Stock Analysis Assistant.
                                
                For news queries, decode the news sentiment ranging from +10 to -10 in table format with columns: Date, Time, News, Source and Score.
                Include the current stock price, key metrics, and recent price movement when a company is mentioned.
                Always provide reasoning for sentiment scores point by point after the tables.
                Always present financial data in tables for clarity.
                Always use tables to display data.                           
                Analyze the query and assign tasks to specialized agents based on their expertise.

                Cite sources with URLs and maintain clarity.
                
                Compile and format all findings from your team into a professional report.\
            """),

            expected_output=dedent("""\
                                   
            # {Compelling Headline}
        
            Research conducted by Financial Agent -
            Published: {current_date}
                                                  
            ---
                                   
            """),
            show_tool_calls=True,
            success_criteria="The user's query has been thoroughly answered with information from all relevant specialists.",
            add_datetime_to_instructions=True,
            #enable_agentic_context=True,
            share_member_interactions=True,
            show_members_responses=False,
            enable_team_history=True, 
            num_of_interactions_from_history=5,
            markdown=True
        )

# Streamlit UI

# Initialize chat history in session state if it doesn't exist
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to analyze?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Add assistant message placeholder
    with st.chat_message("assistant"):
        team = initialize_team()
        if team:
            message_placeholder = st.empty()
            response_text = ""
            
            try:
                response_stream = team.run(prompt, stream=True)
                for chunk in response_stream:
                    response_text += chunk.content
                    message_placeholder.markdown(response_text + "▌")
                
                message_placeholder.markdown(response_text)
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response_text})
            except Exception as e:
                st.error(f"Error generating analysis: {str(e)}")

# Sidebar content
with st.sidebar:
    
    # Example queries in expandable sections
    st.sidebar.header("Example Queries")
    
    with st.expander("📈 Stock Analysis"):
        st.markdown("""
        - What's the latest news and financial performance of Apple (AAPL)?
        - Show me the 52-week high/low for major banking stocks
        - Search for recent insider trading activity at Microsoft
        """)
    
    with st.expander("🔍 Market Research"):
        st.markdown("""
        - Perform a comparable companies analysis for Five Below. Identify its main competitors and market positioning.
        - Which stocks are trending today?
        - What factors are driving oil prices this week?
        """)
    
    with st.expander("📰 News & Sentiment"):
        st.markdown("""
        - Compare the sentiment of news about NVIDIA versus AMD
        - Analyze the sentiment of recent Fed announcements on interest rates
        - What is the latest news about Tesla and how might it affect the stock?
        """)
    
    with st.expander("🏢 Industry Analysis"):
        st.markdown("""
        - Analyze semiconductor industry news and stock performance over the past week
        - What impact did the latest Apple product announcement have on their stock?
        - Is there a correlation between recent Tesla news sentiment and stock price movements?
        """)
    st.sidebar.header("Chat History")
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

try:
    _ = st.secrets["GEMINI_API_KEY"]
except KeyError:
    st.sidebar.warning("⚠️ API KEY is not set in secrets")
