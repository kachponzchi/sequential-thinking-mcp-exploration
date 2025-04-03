import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI

from dotenv import load_dotenv
load_dotenv()

import os, time
from langchain_openai import ChatOpenAI


def get_llms(model_name: str="gpt-3.5-turbo"):
    """
    Helper function to get OpenAI model instance with necessary API key and endpoint.
    
    Args:
        model_name: The name of the model to use
        
    Returns:
        A configured ChatOpenAI instance
    """
    return ChatOpenAI(
        openai_api_key=os.getenv(
            "OPEN_ROUTER_API_KEY", 
            "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"
        ),
        openai_api_base=os.getenv(
            "OPEN_ROUTER_BASE_URL", 
            "https://openrouter.ai/api/v1/chat/completions"
        ),
        model=model_name,
        temperature=0,
        base_url="https://openrouter.ai/api/v1",
        streaming=True
    )

async def main():
    start_time = time.time()
    model = get_llms()
    print(f"Waktu yang dibutuhkan untuk inisiasi model: {time.time() - start_time}")

    async with MultiServerMCPClient(
        {
            "fetch": {
                # Ensure your weather server runs on port 8080
                "url": "http://localhost:8080/sse",
                "transport": "sse",
            }
        }
    ) as client:
        start_time = time.time()
        agent = create_react_agent(model, client.get_tools())
        print(f"Waktu yang dibutuhkan untuk inisiasi agent: {time.time() - start_time}")
        
        # Test query
        query = {"messages": "Do you know about quantum computing?"}
        
        # Invoke agent
        start_time = time.time()
        response = await agent.ainvoke(query)
        print(f"Response time: {time.time() - start_time}")
        print(response)
        
if __name__ == "__main__":
    asyncio.run(main())

    
