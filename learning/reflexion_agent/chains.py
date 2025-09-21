import datetime
import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from schema import AnswerQuestion, ReviseAnswer
from langchain_core.output_parsers.openai_tools import PydanticToolsParser, JsonOutputToolsParser
from langchain_core.messages import HumanMessage

load_dotenv()

pydantic_parser = PydanticToolsParser(tools=[AnswerQuestion])

parser = JsonOutputToolsParser(return_id=True)

actor_prompt_template = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are expert AI researcher.\n"
        "Current time: {time}\n\n"
        "1. {first_instruction}\n"
        "2. Reflect and critique your answer. Be severe to maximize improvement.\n"
        "3. After the reflection, **list 1-3 search queries separately** for researching improvements. Do not include them inside the reflection."
    ),
    MessagesPlaceholder(variable_name="messages"),
    SystemMessagePromptTemplate.from_template("Answer the user's question above using the required format."),
])

actor_prompt_template = actor_prompt_template.partial(
    time=datetime.datetime.now().isoformat()
)

first_responder_prompt_template = actor_prompt_template.partial(
    first_instruction="Provide a detailed ~250 word answer"
)

llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

first_responder_chain = first_responder_prompt_template | llm.bind_tools(tools=[AnswerQuestion], tool_choice='AnswerQuestion' )

validator = PydanticToolsParser(tools=[AnswerQuestion])

revise_instructions = """Revise your previous answer using the new information.
    - You should use the previous critique to add important information to your answer.
        - You MUST include numerical citations in your revised answer to ensure it can be verified.
        - Add a "References" section to the bottom of your answer (which does not count towards the word limit). In form of:
            - [1] https://example.com
            - [2] https://example.com
    - You should use the previous critique to remove superfluous information from your answer and make SURE it is not more than 250 words.
"""

revisor_chain = actor_prompt_template.partial(
    first_instruction = revise_instructions
) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")

response = first_responder_chain.invoke({
    "messages": [HumanMessage(content="Write me a blog post on how small business can leverage AI to grow")]
})

print(response)
