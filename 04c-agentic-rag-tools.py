from typing import Optional, List
from pydantic_ai import Agent, RunContext, ModelRetry
from pydantic import BaseModel, Field
from dataclasses import dataclass
import logfire

from dotenv import load_dotenv

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


## Intent Extraction Agent
@dataclass
class Intent:
  intent: str = Field(..., description='The intent of the user input.')


intent_agent = Agent('gpt-4o-mini',
                     output_type=Intent,
                     retries=5,
                     system_prompt='Extract the intent from the user input.')


@intent_agent.output_validator
async def validate_intent(intent: Intent):
  if intent.intent not in possible_intents:
    raise ModelRetry(
        f'Invalid intent: {intent.intent}. Intent must be one of {possible_intents}.'
    )
  return intent


## Task Summary Agent
class AllTasks(BaseModel):
  tasks: dict = Field(..., description='The tasks.')


class TaskSummary(BaseModel):
  summary: str = Field(..., description='The summary of the tasks.')


summary_agent = Agent('gpt-4o-mini',
                      output_type=TaskSummary,
                      deps_type=AllTasks,
                      retries=5,
                      system_prompt='Summarize the tasks.')


@summary_agent.system_prompt
async def summary_prompt(ctx: RunContext[AllTasks]):
  return f'All Tasks in the database: {ctx.deps.tasks}.'


## Task Extractor Agent
class TaskDetails(BaseModel):
  task_name: str = Field(..., description='The name of the task.')
  task_description: str = Field(...,
                                description='The description of the task.')
  task_done: bool = Field(..., description='Whether the task is done.')


extractor_agent = Agent(
    'gpt-4o-mini',
    output_type=TaskDetails,
    deps_type=AllTasks,
    retries=5,
    system_prompt='Extract the task details from the user input.')


@extractor_agent.system_prompt
async def extractor_prompt(ctx: RunContext[AllTasks]):
  return f'All Tasks in the database: {ctx.deps.tasks}.'


## Finaliser agent
finaliser_agent = Agent(
    model='openai:gpt-4o',
    deps_type=AllTasks,
    retries=5,
    system_prompt=
    ('You are a helpful AI agent that finalizes the response based on the extracted information and user query. '
     'First find intent and then use the corresponding tool. Before adding, call extract task first. Provide a well-formatted response based on the context provided.'
     ),
)


@finaliser_agent.system_prompt
def add_context(ctx: RunContext[AllTasks]):
  return f"All Tasks in the database: {ctx.deps.tasks}"


@finaliser_agent.tool
def extract_task(ctx: RunContext[AllTasks], query: str) -> TaskDetails:
  task_details = extractor_agent.run_sync(query, deps=AllTasks(tasks=tasks_db))
  return task_details.output


@finaliser_agent.tool
def add_task(ctx: RunContext[AllTasks], task_name: str, task_description: str,
             task_done: bool):
  tasks_db[task_name] = {"description": task_description, "done": task_done}
  return f"Task {task_name} added with description {task_description} and done status {task_done}."


@finaliser_agent.tool
def read_tasks(ctx: RunContext[AllTasks], query: str):
  result = summary_agent.run_sync(query, deps=AllTasks(tasks=tasks_db))
  return result.output.summary


@finaliser_agent.tool
def mark_done(ctx: RunContext[AllTasks], task_name: str):
  tasks_db[task_name]["done"] = True
  return f"Task {task_name} marked as done."


query = "Add task to do laundry and add task to call mom and then show all tasks"
final_response = finaliser_agent.run_sync(query, deps=AllTasks(tasks=tasks_db))

print("Final Response:", final_response.output)
