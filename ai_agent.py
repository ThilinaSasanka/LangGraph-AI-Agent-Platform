#step1 : Setup API keys for Groq and Tavily
import os

GROQ_API_KEY=os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY=os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY")

#step2 : Setup LLM and Tools
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

openai_llm=ChatOpenAI(model="gpt-4o-mini")
groq_llm=ChatGroq(model="llama-3.3-70b-versatile")

search_tool=TavilySearchResults(max_results=2)


#step3 : Setup AI agent with Search tool functionality
from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage

system_prompt="Act as an AI chatbot who is smart and friendly"

def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    if llm_id == "gpt-4o-mini":
        llm = ChatOpenAI(model="gpt-4o-mini")
    elif llm_id == "llama-3.3-70b-versatile":
        llm = ChatGroq(model="llama-3.3-70b-versatile")
    elif llm_id == "mixtral-8x7b-32768":
        llm = ChatGroq(model="mixtral-8x7b-32768")
    else:
        raise ValueError("Unsupported llm_id: " + str(llm_id))
    tools = [TavilySearchResults(max_results=2)] if allow_search else []
    agent = create_react_agent(model=llm, tools=tools)
        # Construct messages with system prompt as the first message
    state = {"messages": query}

    response = agent.invoke(state)
    messages = response.get("messages")
    ai_messages = [message.content for message in messages if isinstance(message, AIMessage)]
    return ai_messages[-1]
