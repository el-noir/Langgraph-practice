from langgraph.graph import StateGraph, START
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict,Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
import asyncio
import os
from langchain_mcp_adapters.client import MultiServerMCPClient

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash",google_api_key=os.getenv("GOOGLE_API_KEY")
)

client = MultiServerMCPClient(
    {
        "arithmatic-tool": {
            "transport": "stdio",
            "command": "python",
            "args": ["D:\AI\Langgraph\Langgraph-Learning\learning2\mcp-client\server.py"]
        } 
    }
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

async def build_graph():

    tools = await client.get_tools()
    print(tools)

    llm_with_tools = llm.bind_tools(tools)

    async def chat_node(state: ChatState):
        messages = state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        return {"messages": messages + [response]}
    
    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)

    graph.add_edge("tools", "chat_node")

    chatbot = graph.compile()

    return chatbot

async def main():
    chatbot = build_graph()

    result = await chatbot.ainvoke({"messages": [HumanMessage(content="What is 15 multiplied by 3 and give answer like a philosopher")]})

    print(result["messages"][-1].content)

if __name__ == "__main__":
    asyncio.run(main())