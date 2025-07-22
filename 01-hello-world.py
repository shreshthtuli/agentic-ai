
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai import Agent

from dotenv import load_dotenv

load_dotenv()

model = OpenAIModel(model_name='gpt-4o')
agent = Agent(model)

result = agent.run_sync('Where does "hello world" come from?')  
print(result.output)
