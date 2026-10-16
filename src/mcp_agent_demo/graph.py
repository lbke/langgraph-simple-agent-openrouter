"""Minimal LangChain agent graph for deployment."""

from __future__ import annotations
import asyncio
from typing import TypedDict

import os
from datetime import datetime, timezone

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import Runtime
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

from langchain.mcp import MCPAdapter
from pathlib import Path

# 1. Helper function to fetch MCP tools synchronously using asyncio.run()

# FIXME: proposed by chat.langchain to address langchain being async
# But it doesn't work, we'll wait for official support


def get_mcp_tools():
    mcp_script = Path(__file__).resolve().with_name("mcp_calculator.py")

    async def _fetch():
        async with MCPAdapter(mcp_script) as adapter:
            return await adapter.list_tools()

    return asyncio.run(_fetch())


# For OpenRouter, we won't use this default setup at all
# DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

# https://openrouter.ai/docs/guides/community/langchain
model = init_chat_model(
    # model="mistralai/mistral-small-2603",
    model="google/gemini-3.1-flash-lite",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Contexte = valeurs globales (souvent read-only)


class Context(TypedDict):
    pass

# State = état de l'agent

# Important : doit hériter de "AgentState" pour le champ messages


class State(AgentState):
    pass

# Autres exempltes d'outils


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


mcp_tools = get_mcp_tools()
graph = create_agent(
    model=model,  # DEFAULT_MODEL,
    tools=[utc_now, *mcp_tools],
    state_schema=State,
    context_schema=Context,
    system_prompt=(
        "You are a concise assistant. "
        "Use tools when they add factual precision, then return a direct answer."
    ),
    name="simple_agent",
)
