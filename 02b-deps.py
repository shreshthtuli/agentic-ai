
from pydantic_ai import Agent, RunContext
from pydantic import BaseModel, Field
from typing import List

from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

class Order(BaseModel):
    order_id: str 
    status: str 
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

agent = Agent(  
    'openai:gpt-4o',
    deps_type=CustomerDetails,
    output_type=ResponseModel,
    system_prompt=(
        'You are an intelligent customer support agent. '
        'Analyze the customer details and provide structured responses.'
    ),
)

# Dynamic system prompt to include customer details
@agent.system_prompt
async def add_customer_details(ctx: RunContext[CustomerDetails]):
    return f"Customer Details: {ctx.deps.model_dump_json(indent=2)}"


result = agent.run_sync('What did I order?', deps=CustomerDetails(
    customer_id="12345",
    name="John Doe",
    email="john.doe@example.com",
    orders=[
        Order(order_id="1", status="shipped", items=["item1", "item2"]),
        Order(order_id="2", status="processing", items=["item3"]),
    ]
))
result.all_messages()
print(result.output.model_dump_json(indent=2))

