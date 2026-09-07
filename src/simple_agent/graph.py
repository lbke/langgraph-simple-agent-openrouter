"""Minimal LangChain agent graph for deployment."""

from __future__ import annotations
from typing import TypedDict

import os
from datetime import datetime, timezone

from langchain.agents import AgentState, create_agent
from langchain.agents.middleware import Runtime
from langchain.chat_models import init_chat_model
from langchain.tools import ToolRuntime
from langchain_core.tools import tool

# For OpenRouter, we won't use this default setup at all
# DEFAULT_MODEL = os.getenv("SIMPLE_AGENT_MODEL", "anthropic:claude-sonnet-4-6")

# https://openrouter.ai/docs/guides/community/langchain
model = init_chat_model(
    model="mistralai/mistral-small-2603",
    model_provider="openai",
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

# Contexte = valeurs globales (souvent read-only)


class Context(TypedDict):
    nom_utilisateur: str

# State = état de l'agent

# Important : doit hériter de "AgentState" pour le champ messages


class State(AgentState):
    aime_python: bool | None
    aime_javascript: bool | None


@tool
def read_preference(runtime: ToolRuntime[Context]):
    """Read state"""
    return runtime.state.get("aime_python")


@tool
def update_preference(runtime: ToolRuntime[Context], aime_python: bool):
    """Update state"""
    runtime.state["aime_python"] = aime_python
    return "Préférence mise à jour avec succès"

# Autres exempltes d'outils


@tool
def utc_now() -> str:
    """Return the current UTC timestamp in ISO format."""
    return datetime.now(tz=timezone.utc).isoformat()


graph = create_agent(
    model=model,  # DEFAULT_MODEL,
    tools=[utc_now, read_preference, update_preference],
    state_schema=State,
    context_schema=Context,
    system_prompt=(
        "You are a concise assistant. "
        "Use tools when they add factual precision, then return a direct answer."
    ),
    name="simple_agent",
)
