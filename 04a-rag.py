from typing import Optional
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import logfire

from src.utils import perform_vector_search, perform_fts_search, build_context_from_results, read_files_as_object_array

from dotenv import load_dotenv
import nest_asyncio
nest_asyncio.apply()
load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()



ollama_model = OpenAIModel(
    model_name='qwen3:32b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'))
agent = Agent(
    model=ollama_model,
    retries=5,
    system_prompt=(
        'You are a helpful AI assistant. '
        'Help get the right information using the vector similarity search and keyword search capabilities. '
    ),
)

@agent.system_prompt
async def add_customer_details(ctx: RunContext[None]):
    files = read_files_as_object_array('data/')
    return "Here's the list of pokemons you can search for:\n" + \
        "\n".join([f"{file['filename'].split('.')[0]}" for file in files])

@agent.tool_plain
def perform_similarity_search(query: str, pokemon: Optional[str]) -> list[str]:
    print(f"Performing similarity search for query: {query}, pokemon: {pokemon}")
    results = perform_vector_search(query, pokemon=pokemon+".md", top_k=5)
    return build_context_from_results(results)

@agent.tool_plain
def perform_keyword_search(query: str, pokemon: Optional[str]) -> list[str]:
    print(f"Performing keyword search for query: {query}, pokemon: {pokemon}")
    results = perform_fts_search(query, pokemon=pokemon+".md", top_k=5)
    return build_context_from_results(results)


result = agent.run_sync("What are the electric-type moves of Pikachu?")
result.all_messages()
print(result.output)

result = agent.run_sync("Who has more powerful normali type attack - Charizard or Pikachu?")
result.all_messages()
print(result.output)
