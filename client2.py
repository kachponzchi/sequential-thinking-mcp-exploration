import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnableConfig
from langchain_core.callbacks import BaseCallbackHandler
import json

import os
from langchain_openai import ChatOpenAI

def invoke(self, input_query):
    inputs = {"question": input_query}
    result = None
    config = RunnableConfig(recursion_limit=50)
    try:
        for output in self.app.stream(inputs, config):
            for key, value in output.items():
                logging.info(f"Node '{key}':")
                result = value
    except GraphRecursionError:
        result = None
        logging.error("Graph recursion limit reached.")

    return result


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
    
    model = get_llms()

    async with MultiServerMCPClient(
        {
            "linear": {
                "url": "http://localhost:8080/sse",
                "transport": "sse",
            }
        }
    ) as client:
        # First, let's get and display available states
        tools = client.get_tools()
        print("\nAvailable Linear tools and their descriptions:")
        for tool in tools:
            print(f"\nTool: {tool.name}")
            print(f"Description: {tool.description}")
            if hasattr(tool, 'args'):
                print("Arguments:", json.dumps(tool.args, indent=2))
        
        agent = create_react_agent(model, tools)
        
        while True:
            # Get user input
            user_input = input("\nEnter your message (or 'quit' to exit): ")
            
            # Check for quit condition
            if user_input.lower() == 'quit':
                break
            
            # Create the query
            query = {"messages": user_input}
            
            try:
                # Create a custom callback handler
                class RecursionTracker(BaseCallbackHandler):
                    def __init__(self):
                        super().__init__()
                        self.count = 0
                    
                    def on_llm_start(self, *args, **kwargs):
                        self.count += 1
                        print(f"\nRecursion step {self.count}")

                tracker = RecursionTracker()
                config = RunnableConfig(recursion_limit=50, callbacks=[tracker])
                response = await agent.ainvoke(query, config)
                
                # Print final recursion count
                print(f"\nTotal recursions: {tracker.count}")
                
                # Extract and print the AI's response
                messages = response.get('messages', [])
                for message in messages:
                    if hasattr(message, 'content'):
                        print("\nAI Response:")
                        print("-" * 50)
                        print(message.content)
                        print("-" * 50)
            except Exception as e:
                print(f"\nError: {str(e)}")
                print("Please try again or type 'quit' to exit.")

if __name__ == "__main__":
    asyncio.run(main())