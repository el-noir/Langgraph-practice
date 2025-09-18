from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from langchain.agents import initialize_agent
from langchain.tools import TavilySearchResults   # 👈 old import

load_dotenv()

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

search_tool = TavilySearchResults(search_depth="basic")  # 👈 old class

tools = [search_tool]

agent = initialize_agent(tools=tools, llm=llm, agent="zero-shot-react-description", verbose=True)

agent.invoke("Give me a funny tweet about today's weather in Bangalore")
