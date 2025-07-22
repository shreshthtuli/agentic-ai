
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
import requests
import logfire

from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()


class ResponseModel(BaseModel):
    response: str
    sentiment: str = Field(description="Customer Sentiment Analysis", default=None)

ollama_model = OpenAIModel(
    model_name='qwen3:32b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'))
agent = Agent(
    model=ollama_model,
    output_type=ResponseModel,
    retries=5,
    system_prompt=(
        'You are an intelligent customer support agent. '
        'Use web search to answer user questions. '
    ),
)

@agent.tool_plain
def search(query: str) -> list[str]:
    url = f"https://duckduckgo.com/?q={query}&format=xml"
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'xml')
    results = []
    for result in soup.find_all('result'):
        title = result.find('title').text
        url = result.find('url').text
        results.append(f"{title}: {url}")
    return results

result = agent.run_sync("what is the weather in New York today?")
result.all_messages()
print(result.output.model_dump_json(indent=2))

