
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
from typing import List
import logfire

from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()

class Order(BaseModel):
    order_id: str 
    items: List[str] = Field(description="List of items in the order", default=[])

class CustomerDetails(BaseModel):
    customer_id: str 
    name: str 
    email: str 
    orders: List[Order] = Field(description="List of orders placed by the customer", default=[])

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
    deps_type=CustomerDetails,
    output_type=ResponseModel,
    retries=5,
    system_prompt=(
        'You are an intelligent customer support agent. '
        'Analyze the customer details and provide structured responses using tools. '
    ),
)

@agent.tool_plain
async def get_shipping_status(order_id: str) -> str:
    statuses = {
        '#1': "shipped",
        '#2': "processing",
        '#3': "delivered",
        '#4': "cancelled"
    }
    if order_id not in statuses:
        raise ModelRetry(f"Order ID {order_id} not found. It should start with '#'.")

    return statuses[order_id]


# Dynamic system prompt to include customer details
@agent.system_prompt
async def add_customer_details(ctx: RunContext[CustomerDetails]):
    return f"Customer Details: {ctx.deps.model_dump_json(indent=2)}"


result = agent.run_sync('What are the statuses of all my orders?', deps=CustomerDetails(
    customer_id="12345",
    name="John Doe",
    email="john.doe@example.com",
    orders=[
        Order(order_id="1", items=["item1", "item2"]),
        Order(order_id="2", items=["item3"]),
    ]
))
result.all_messages()
print(result.output.model_dump_json(indent=2))

