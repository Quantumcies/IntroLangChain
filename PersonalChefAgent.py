import dotenv #load environment variables
dotenv.load_dotenv()
import langchain #agent library
import base64 #multimodal encoding
from langgraph.checkpoint.memory import InMemorySaver #memory
from typing import Dict, Any, List #type hints
from tavily import TavilyClient #agent websearch
from langchain.agents import create_agent #agent
from langchain.chat_models import init_chat_model #agent
from langchain.tools import tool #agent tools
from langchain.messages import HumanMessage #prompts
from langchain.messages import AIMessage #agent responses
from pydantic import BaseModel #prompt engineering
from pprint import pprint #pretty print

# Initialize Clients
tavily_client = TavilyClient()

#Agent Environment
system_prompt="You are a world class personal chef, you expertise is in creative receipe development on budget ingredients. Return receipe suggestions and eventually the instructions to the user."
question = HumanMessage(content="Apples, cranberries, flour, eggs, butter, sugar, salt, vanilla extract, heavy whipping cream")

class ChefsInstructions(BaseModel):
    ingredients: List[str]
    receipe: str
    steps: List[str]

@tool
def receipe_search(query: str) -> Dict[str, Any]:
    """Search the web for information."""
    return tavily_client.search(query)

@tool
def image_recognition(image_path: str) -> str:
    """Reads a local image file and returns the base64 string."""
    try:
        with open(image_path, "rb") as image_file:
            img_bytes = image_file.read()
            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
            return img_b64
    except FileNotFoundError:
        return f"Error: Image file not found at {image_path}"

#===========CREATE AGENT========================
model = init_chat_model(model="gpt-5-nano", temperature=1, timeout=60, max_retries=2)
agent = create_agent(
    model,
    system_prompt=system_prompt,
    response_format=ChefsInstructions,
    tools=[receipe_search, image_recognition],
    checkpointer=InMemorySaver(),
)

#===========STREAM RESPONSE=====================
print("Invoking agent...")

# Check if default image exists, otherwise warn user
import os
default_image = "ingredients.jpg"
if not os.path.exists(default_image):
    print(f"Note: '{default_image}' not found. Place an image with this name in the folder to test image recognition.")
    initial_message = "I have some ingredients in the kitchen."
else:
    initial_message = f"I have some ingredients in the kitchen. Please look at {default_image}"

# We pass the question directly in the messages list
for token, metadata in agent.stream(
    {"messages": [HumanMessage(content=initial_message), question]},
    config={"configurable": {"thread_id": "1"}}, # Added required checkpointer config
    stream_mode="messages"
):
    if token.content: #check actual content
        print(token.content, end="", flush=True) #print token