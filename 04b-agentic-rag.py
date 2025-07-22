from typing import Optional
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic import BaseModel, Field
from typing import List
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

# ollama_model = OpenAIModel('gpt-4o')

### Planner Agent
class Subquestions(BaseModel):
    subquestions: List[str]

planner_agent = Agent(
    model=ollama_model,
    output_type=Subquestions,
    retries=5,
    system_prompt=(
        'You are a helpful AI agent that plans a query into subqueries. '
        'You need to decompose the user query into subquestions where each subquestion can be answered independently. '
        'Try to keep the subquestions as few as possible while ensuring they cover the original query comprehensively. '
    ),
)

@planner_agent.system_prompt
async def add_pokemon_list(ctx: RunContext[None]):
    files = read_files_as_object_array('data/')
    return "Here's the list of all pokemons:\n" + \
        "\n".join([f"{file['filename'].split('.')[0]}" for file in files])

@planner_agent.output_validator
async def validate_subquestions(subquestions: Subquestions) -> bool:
    if not subquestions.subquestions:
        raise ModelRetry("Subquestions list cannot be empty.")
    return subquestions


## Retriever Agent
retriever_agent = Agent(
    model=ollama_model,
    retries=5,
    system_prompt=(
        'You are a helpful AI assistant. '
        'Help get the right information using the vector similarity search and keyword search capabilities. '
    ),
)

@retriever_agent.system_prompt
async def add_pokemon_list(ctx: RunContext[None]):
    files = read_files_as_object_array('data/')
    return "Here's the list of pokemons you can search for:\n" + \
        "\n".join([f"{file['filename'].split('.')[0]}" for file in files])

@retriever_agent.tool_plain
def perform_similarity_search(query: str, pokemon: str) -> list[str]:
    print(f"Performing similarity search for query: {query}, pokemon: {pokemon}")
    results = perform_vector_search(query, pokemon=pokemon+".md", top_k=5)
    return build_context_from_results(results)

@retriever_agent.tool_plain
def perform_keyword_search(query: str, pokemon: str) -> list[str]:
    print(f"Performing keyword search for query: {query}, pokemon: {pokemon}")
    results = perform_fts_search(query, pokemon=pokemon+".md", top_k=5)
    return build_context_from_results(results)


## Extractor Agent
class ExtractorInput(BaseModel):
    context: str

extractor_agent = Agent(
    model=ollama_model,
    retries=5,
    deps_type=ExtractorInput,
    system_prompt=(
        'You are a helpful AI assistant. '
        'Extract the relevant information from the context provided for the given query. '
    ),
)

@extractor_agent.system_prompt
async def add_context(ctx: RunContext[ExtractorInput]):
    return f"Context: {ctx.deps.context}"

## Finaliser Agent
class FinaliserInput(BaseModel):
    extracted_info: List[str]

finaliser_agent = Agent(
    model=ollama_model,
    retries=5,
    deps_type=FinaliserInput,
    system_prompt=(
        'You are a helpful AI assistant. '
        'Combine the extracted information from multiple subquestions to provide a comprehensive answer to the original query. '
    ),
)

@finaliser_agent.system_prompt
async def add_context(ctx: RunContext[FinaliserInput]):
    return f"Context: {ctx.deps.extracted_info}"


### Example Usage

# query = "Who has more powerful normal type attack - Charizard or Pikachu?"
query = "Name the pokemons that learn the moves thunderbolt or growl."

subquestions = planner_agent.run_sync(query)
print("Subquestions:")
print(subquestions.output.model_dump_json(indent=2))

contexts = []
for subquestion in subquestions.output.subquestions:
    print(f"Retrieving context for subquestion: {subquestion}")
    retrieved_context = retriever_agent.run_sync(subquestion)
    print(retrieved_context.output)

    prompt_for_extractor = ExtractorInput(
        context=retrieved_context.output
    )
    extracted_info = extractor_agent.run_sync(subquestion, deps=prompt_for_extractor)
    print(f"Extracted Information for '{subquestion}': {extracted_info.output}")

    contexts.append(retrieved_context.output)

final_input = FinaliserInput(
    query=query,
    extracted_info=contexts
)
final_result = finaliser_agent.run_sync(query, deps=final_input)
print("Final Result:")
print(final_result.output)

