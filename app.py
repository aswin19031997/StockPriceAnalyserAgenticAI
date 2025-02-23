import streamlit as st
import pandas as pd 
import dotenv
from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
import io
import contextlib
import json

# Initialize Agents
web_search_agent = Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],

)

finance_agent = Agent(
    name="Finance AI Agent",
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    tools=[YFinanceTools(stock_price=True, analyst_recommendations=True, stock_fundamentals=True, company_news=True)],
    instructions=["Use tables to display the data"],

)

multi_ai_agent = Agent(
    model=Groq(id="deepseek-r1-distill-llama-70b"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources and use table to display data"],

)

def display_result(result):

    if isinstance(result, str):
        result_data = json.loads(result)
    elif hasattr(result, "content"):
        result_data = {"content": result.content}
    else:
        result_data = {}

    content = result_data.get("content", "No content available.")
    st.markdown("### Summary")
    st.markdown(content)


# Streamlit UI
st.title("Agentic AI App for Stock Analysis")
st.write("This app allows you to query different AI agents for web search or financial data.")

# Agent selection
agent_option = st.selectbox("Select Agent", ["Web Search", "Finance", "Multi"])

# Input area for the query
user_query = st.text_area("Enter your query", placeholder="Type your query here...")

if st.button("Submit Query"):
    if user_query.strip():
        st.write("Processing your query...")
        try:
            with st.spinner("Waiting for agent response..."):
                if agent_option == "Web Search":
                    query = f"Summarize analyst recommendations and share the latest news for {user_query}"
                    result = web_search_agent.run(query)
                    st.subheader("Web Search Agent Result")
                    display_result(result)
                elif agent_option == "Finance":
                    result = finance_agent.run(user_query)
                    st.subheader("Finance AI Agent Result")
                    display_result(result)
                else:
                    result = multi_ai_agent.run(user_query)
                    st.subheader("Multi AI Agent Result")
                    display_result(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a valid query before submitting.")
