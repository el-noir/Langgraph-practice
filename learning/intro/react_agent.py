# from langchain_google_genai import ChatGoogleGenerativeAI
# from dotenv import load_dotenv
# from langchain.agents import initialize_agent, tool
# from langchain_tavily import TavilySearch   # ✅ new import
# import datetime

# load_dotenv()

# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# # ✅ New class instead of TavilySearchResults
# search_tool = TavilySearch(search_depth="basic")

# @tool
# def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
#     """
#     Returns the current system time formatted as a string.

#     Args:
#         format (str): Format string for datetime (default: "%Y-%m-%d %H:%M:%S").

#     Returns:
#         str: The formatted current system time.
#     """
#     current_time = datetime.datetime.now()
#     formatted_time = current_time.strftime(format)
#     return formatted_time

# tools = [search_tool, get_system_time]

# agent = initialize_agent(
#     tools=tools,
#     llm=llm,
#     agent="zero-shot-react-description",
#     verbose=True
# )

# response = agent.invoke("When was SpaceX's last launch and how many days ago was that from this instant?")
# print(response)

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain_tavily import TavilySearch
from langchain.agents import tool
from langgraph.prebuilt import create_react_agent
import datetime

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

# ✅ New TavilySearch (multi-input tool supported in LangGraph)
search_tool = TavilySearch()

@tool
def get_system_time(format: str = "%Y-%m-%d %H:%M:%S") -> str:
    """
    Returns the current system time formatted as a string.
    """
    current_time = datetime.datetime.now()
    return current_time.strftime(format)

tools = [search_tool, get_system_time]

# ✅ Use LangGraph’s prebuilt ReAct agent
agent = create_react_agent(llm, tools)

# Run the agent
# response = agent.invoke({"messages": [("user", "When was SpaceX's last launch and how many days ago was that from this instant?")]})
# print(response)

response = agent.invoke(
    {"input": "When was SpaceX's last launch and how many days ago was that?"}
)

print("\n--- ReAct Trace ---\n")
for m in response["messages"]:
    if m["type"] == "ai":
        print(f"🤔 Thought/AI: {m['content']}\n")
    elif m["type"] == "tool":
        print(f"🔧 Tool Call: {m['tool']} with {m['tool_input']}")
        print(f"📩 Observation: {m['content']}\n")

print("\n--- Final Answer ---\n")
print(response["messages"][-1]["content"])
