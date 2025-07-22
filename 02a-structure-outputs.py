
from pydantic_ai import Agent, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
import logfire

from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()


class ResponseModel(BaseModel):
    response: str
    needs_escalation: bool 
    followup_required: bool
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
        'Your task is to assist customers with their inquiries.'
    ),
)

@agent.output_validator
def validate_response(response: ResponseModel) -> bool:
    if response.sentiment not in ['positive', 'neutral', 'negative']:
        raise ModelRetry("Sentiment must be one of: positive, neutral, negative.")
    return response

response = agent.run_sync('How can I track my order?')
print(response.output)

