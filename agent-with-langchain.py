import dotenv
import os
from typing import Annotated, TypedDict

# ==================== LANGCHAIN IMPORTS ====================
# LangChain = building blocks: LLM wrappers, tools, message types
# LangChain: Groq LLM integration
from langchain_groq import ChatGroq
# LangChain: turns a Python function into a tool
from langchain_core.tools import tool
# LangChain: chat message types
from langchain_core.messages import HumanMessage, SystemMessage
# NEW - LangChain: ready-made Tavily web search tool
from langchain_tavily import TavilySearch

# ==================== LANGGRAPH IMPORTS ====================
# LangGraph = orchestration: state, nodes, edges, loops
# LangGraph: graph builder + entry point
from langgraph.graph import StateGraph, START
# LangGraph: reducer that appends messages to state
from langgraph.graph.message import add_messages
# LangGraph: ready-made tool node + routing logic
from langgraph.prebuilt import ToolNode, tools_condition

dotenv.load_dotenv()


# ============================================================
# LANGCHAIN SECTION: LLM + Tools
# Defines WHAT the agent can use: the model and the functions it may call.
# ============================================================

# LangChain: wrap the Groq-hosted model in a standard chat model interface
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ.get("GROQ_API_KEY"),
    temperature=0.5)


# LangChain: @tool reads the function name, type hints and docstring
# so the LLM knows what the tool does and which arguments it takes
@tool
def get_capital(country: str) -> str:
    """Return the capital city of a country."""
    capitals = {"france": "Paris", "japan": "Tokyo",
                "bangladesh": "Dhaka", "usa": "Washington, D.C."}
    return capitals.get(country.lower(), f"I don't know the capital of {country}.")


@tool
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


# NEW - LangChain: Tavily web search tool
# It reads TAVILY_API_KEY from .env automatically.
# max_results=3 keeps the results short (fewer tokens sent back to the LLM).
web_search = TavilySearch(max_results=3)

# NEW - web_search added to the tool list
tools = [get_capital, multiply, web_search]

# LangChain: bind_tools sends the tool schemas to the LLM with every request,
# so the model can reply with a tool call instead of plain text
llm_with_tools = llm.bind_tools(tools)


# ============================================================
# LANGGRAPH SECTION: State, Nodes, Graph
# (no changes - the graph doesn't care which tools exist)
# ============================================================

# LangGraph: State is the shared data passed between nodes.
# add_messages makes each node's output get APPENDED to the history (not overwritten).
class State(TypedDict):
    messages: Annotated[list, add_messages]


# LangGraph node (it calls a LangChain LLM inside):
# reads the state, asks the LLM, returns the new message to add to the state
def agent(state: State):
    response = llm_with_tools.invoke(state["messages"])   # LangChain call
    # LangGraph state update
    return {"messages": [response]}


# LangGraph: build the workflow
builder = StateGraph(State)

# LangGraph: nodes = the steps
builder.add_node("agent", agent)            # step 1: LLM thinks / decides
# step 2: run the tool(s) the LLM asked for
builder.add_node("tools", ToolNode(tools))

# LangGraph: edges = the flow between steps
# always start at the agent
builder.add_edge(START, "agent")
# LLM asked for a tool? -> "tools", else -> END
builder.add_conditional_edges("agent", tools_condition)
# after the tool runs, back to the LLM (the agent loop)
builder.add_edge("tools", "agent")

# LangGraph: compile into a runnable app
graph = builder.compile()


# ============================================================
# RUN
# graph.invoke is LangGraph; the messages inside are LangChain types
# ============================================================
def main():
    result = graph.invoke({                    # LangGraph: run the whole graph
        "messages": [
            # LangChain message
            # NEW - system prompt tells the agent when to use web search
            SystemMessage(
                "You are a helpful assistant. Use tools when they help. "
                "Use web_search for recent or current information, and include the source URLs."),
            # LangChain message
            # NEW - question that needs a web search
            HumanMessage(
                "What is the latest version of Playwright, and what is 12 times 7?"),
        ]
    })

    # LangChain: pretty_print shows each message (human, AI tool call, tool result, final answer)
    for msg in result["messages"]:
        msg.pretty_print()


if __name__ == "__main__":
    main()
