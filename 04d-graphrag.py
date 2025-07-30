import networkx as nx
from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider
import matplotlib.pyplot as plt


import logfire
import nest_asyncio
nest_asyncio.apply()

from dotenv import load_dotenv

load_dotenv()

logfire.configure()
logfire.instrument_pydantic_ai()

G = nx.DiGraph()

def graph_to_string():
    """Convert the graph to a string representation."""
    result = []
    for edge in G.edges(data=True):
        result.append(f"<Node1> {edge[0]}, <Relation> {edge[2]['relation']}, <Node2> {edge[1]}")
    return "\n".join(result)

def plot_graph():
  plt.figure(figsize=(10, 6))
  pos = nx.spring_layout(G)
  nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_color="black", font_weight="bold", arrows=True)
  nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): d['relation'] for u, v, d in G.edges(data=True)}, font_color='red')
  plt.title("Knowledge Graph")
  plt.show()

## Knowledge Graph Builder Agent
ollama_model = OpenAIModel(
    model_name='qwen3:32b', 
    provider=OpenAIProvider(base_url='http://localhost:11434/v1'))


graph_builder_agent = Agent(
    model='gpt-4o-mini', #ollama_model,
    output_type=None,  # No specific output type, just building the graph
    retries=5,
    system_prompt=(
        'You are a helpful AI agent that builds a knowledge graph from the provided inputs. '
    ),
)

@graph_builder_agent.system_prompt
async def add_graph_context(ctx: RunContext[None]):
    return "Graph: \n " + graph_to_string()

@graph_builder_agent.tool_plain
def add_graph_node(node1, desc=""):
    G.add_node(node1, description=desc)

@graph_builder_agent.tool_plain
def add_graph_edge(node1, node2, edge_desc=""):
    if not G.has_node(node1) or not G.has_node(node2):
        raise ValueError("Both nodes must exist in the graph before adding an edge.")
    G.add_edge(node1, node2, relation=edge_desc)


## Graph Query Agent
graph_query_agent = Agent(
    model='gpt-4o-mini', #ollama_model,
    retries=5,
    system_prompt=(
        'You are a helpful AI agent that answers questions about the knowledge graph. '
    ),
)

@graph_query_agent.system_prompt
async def add_graph_context(ctx: RunContext[None]):
    return "Graph: \n " + graph_to_string()

@graph_query_agent.tool_plain 
def get_adjacent_nodes(node: str) -> str:
    if node in G:
        return str(G[node])
    else:
        return f"Node {node} does not exist in the graph."

@graph_query_agent.tool_plain
def get_node_info(node: str) -> str:
    if node in G:
        return G.nodes[node]['description']
    else:
        return f"Node {node} does not exist in the graph."
    
# Build the graph with some example nodes and edges
result = graph_builder_agent.run_sync(
    "Add a node 'Shray' with description 'Sports heavy user'."
    "Add a node 'Wilko' with description 'Gaming heavy user'. "
    "Add a node 'Kanaad' with description 'VIP user'. "
    "Add a node 'Superbet' with description 'Sports betting company'. "
    "Add an edge from 'Shray' to 'Superbet' with relation 'employee of'. "
    "Add an edge from 'Wilko' to 'Superbet' with relation 'employee of'. "
    "Add an edge from 'Kanaad' to 'Superbet' with relation 'employee of'. "
    "Add an edge from 'Kanaad' to 'Wilko' with relation 'friend of'. ")
print(result.output)

print(graph_to_string())
plot_graph()

# Query the graph
result = graph_query_agent.run_sync(
    "What is the relation between 'Kanaad' and 'Wilko'?")
print(result.output)

result = graph_query_agent.run_sync(
    "What can you tell me about the user who is a friend of 'Wilko'?")
print(result.output)
