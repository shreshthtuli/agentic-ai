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

possible_intents = ['addTask', 'readTasks', 'markDone']

tasks_db = {
    "example task": {
        "description": "This is an example task.",
        "done": False
    }
}

### Intent Extraction Agent
class GetIntent(BaseModel):
    intent: str = Field(description="The intent of the user query: one of " + ", ".join(possible_intents), default=None)

intent_agent = Agent(
    model='openai:gpt-4o',
    output_type=GetIntent,
    retries=5,
    system_prompt=(
        'You are a helpful AI agent that gets the intent of the user query.'
    ),
)

@intent_agent.output_validator
async def validate_intent(intent: GetIntent) -> bool:
    if not intent.intent:
        raise ModelRetry("Intent cannot be empty.")
    if intent.intent not in possible_intents:
        raise ModelRetry(f"Invalid intent: {intent.intent}. Intent must be one of: {possible_intents}.")
    return intent

## Task Summary Agent
class AllTasks(BaseModel):
    tasks: dict = Field(description="All tasks in the database", default=None)

class TaskSummary(BaseModel):
    tasks_summary: str = Field(description="Summary of tasks based on user query", default=None)

summary_agent = Agent(
    model='openai:gpt-4o',
    deps_type=AllTasks,
    output_type=TaskSummary,
    retries=5,
    system_prompt=(
        'You are a helpful AI agent that summarizes tasks based on user queries.'
    ),
)

@summary_agent.system_prompt
async def add_tasks_list(ctx: RunContext[AllTasks]):
    tasks_list = "\n".join([f"{task_name}: {task['description']} (Done: {task['done']})" for task_name, task in ctx.deps.tasks.items()])
    return f"Here are the tasks in the database:\n{tasks_list}"

## Task Extractor Agent
class TaskDetails(BaseModel):
    task_name: str = Field(description="Name of the task to be added or read", default=None)
    task_description: Optional[str] = Field(description="Description of the task to be added", default=None)
    task_done: Optional[bool] = Field(description="Status of the task (done or not done)", default=None)

task_extractor_agent = Agent(
    model='openai:gpt-4o',
    deps_type=AllTasks,
    output_type=TaskDetails,
    retries=5,
    system_prompt=(
        'You are a helpful AI agent that extracts task details from user queries given existing tasks.'
    ),
)

@task_extractor_agent.system_prompt
async def add_tasks_list(ctx: RunContext[AllTasks]):
    tasks_list = "\n".join([f"{task_name}: {task['description']} (Done: {task['done']})" for task_name, task in ctx.deps.tasks.items()])
    return f"Here are the tasks in the database:\n{tasks_list}"

## Finaliser Agent
class FinaliserInput(BaseModel):
    query: str = Field(description="Original user query", default=None)
    extracted_info: List[str] = Field(description="Extracted information from the context", default=[])

finaliser_agent = Agent(
    model='openai:gpt-4o',
    deps_type=FinaliserInput,
    system_prompt=(
        'You are a helpful AI agent that finalizes the response based on the extracted information and user query. '
        'Provide a well-formatted response based on the context provided.'
    ),
)

@finaliser_agent.system_prompt
async def add_context(ctx: RunContext[FinaliserInput]):
    return f"Context: {ctx.deps.extracted_info}"

### Example Usage

query = "Is there a task named 'example task'?" 

def main(query):
    result = intent_agent.run_sync(query)

    context = []
    if result.output.intent == 'addTask':
        task_details = task_extractor_agent.run_sync(query)
        if task_details.output.task_name in tasks_db:
            context.append(f"Task '{task_details.output.task_name}' already exists.")
        else:
            tasks_db[task_details.output.task_name] = {
                "description": task_details.output.task_description or "No description provided.",
                "done": task_details.output.task_done or False
            }
            context.append(f"Task '{task_details.output.task_name}' added successfully.")
    elif result.output.intent == 'readTasks':
        all_tasks = AllTasks(tasks=tasks_db)
        task_summary = summary_agent.run_sync(query, deps=all_tasks)
        context.append(f"Tasks Summary: {task_summary.output.tasks_summary}")
    elif result.output.intent == 'markDone':
        all_tasks = AllTasks(tasks=tasks_db)
        task_details = task_extractor_agent.run_sync(query, deps=all_tasks)
        if task_details.output.task_name in tasks_db:
            tasks_db[task_details.output.task_name]['done'] = True
            context.append(f"Task '{task_details.output.task_name}' marked as done.")
        else:
            context.append(f"Task '{task_details.output.task_name}' not found.")
    else:
        context.append(f"Unknown intent: {result.output.intent}. Please try again with a valid intent.")

    final_result = finaliser_agent.run_sync(query, deps=FinaliserInput(
        extracted_info=context
    ))

    print("Final Result:")
    print(final_result.output)

