import dotenv
from phi.agent import  Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.duckduckgo import DuckDuckGo
import os
from dotenv import load_dotenv
import openai

load_dotenv()
#loading openai key 


#setting openai api key

#Creating an agent for websearch
web_search_agent=Agent(
    name="Web Search Agent",
    role="Search the web for the information",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[DuckDuckGo()],
    instructions=["Always include sources"],

)

## Financial Agent
finance_agent=Agent(
    name="Finance AI Agent",
    model=Groq(id="llama-3.3-70b-versatile"),
    tools=[YFinanceTools(stock_price=True,analyst_recommendations=True,stock_fundamentals=True,company_news=True)],
    instructions=["Use tables to display the data"],

)

multi_ai_agent=Agent(
    model=Groq(id="llama-3.3-70b-versatile"),
    team=[web_search_agent, finance_agent],
    instructions=["Always include sources and use table to display data"],

)

multi_ai_agent.print_response("Summarize anlayst recommendations and share the latest news for soundhound ai",stream=True)
