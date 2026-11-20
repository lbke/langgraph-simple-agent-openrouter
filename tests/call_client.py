from langgraph_sdk import get_client
import asyncio
import os
# Va se connecter à l'agent en local
# Vous pouvez aussi tester avec l'adresse Render
# Ou un tunnel avec : uv run langgraph dev --tunnel
CLIENT_URL = os.environ.get("AGENT_SERVER_URL", "http://127.0.0.1:2024")
client = get_client(url=CLIENT_URL)
# Get stream


async def call():
    print("Available assistants: ", await client.assistants.search())
    async for chunk in client.runs.stream(
        None,  # Threadless run
        "agent",  # Name of assistant. Defined in langgraph.json.
        input={
            "messages": [{
                "role": "human",
                "content": "What is LangGraph?",
            }],
        },
        stream_mode="updates",
    ):
        print(f"Receiving chunk (type: {chunk.event}):")
        print(chunk.data)
        print("\n\n")
if __name__ == "__main__":
    asyncio.run(call())
