import os
from dotenv import load_dotenv
from maxq.search_engine import MaxQEngine, CollectionStrategy, SearchRequest
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

# Initialize Rich Console for output
console = Console()

# Load environment variables
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
    console.print(f"[dim]Loaded configuration from: {env_path}[/dim]")
else:
    # Fallback: Try looking in parent directory explicitly
    parent_env = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
    if os.path.exists(parent_env):
        load_dotenv(parent_env)
        env_path = parent_env
        console.print(f"[dim]Loaded configuration from: {env_path} (parent directory)[/dim]")
    else:
        console.print(f"[dim]No .env file found (checked current and parent directories)[/dim]")

def main():
    # 1. Initialize Engine
    engine = MaxQEngine(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )
    
    # 2. Define Strategy
    config = CollectionStrategy(
        collection_name="awesome-chatgpt-prompts",
        estimated_doc_count=1000,
        use_quantization=True,
        dense_model_name="BAAI/bge-base-en-v1.5"
    )
    
    print(f"Connected to MaxQ Engine. Collection: {config.collection_name}")
    
    # 3. Interactive Search Loop
    print("\n--- MaxQ Search (Type 'exit' to quit) ---")
    while True:
        try:
            query = input("\nQuery: ")
            if query.lower() in ['exit', 'quit']:
                break
                
            request = SearchRequest(
                query=query,
                limit=3,
                strategy="hybrid"
            )
            
            results = engine.query(config, request)
            
            if not results:
                print("No results found.")
                continue
                
            for hit in results:
                print(f"\n[Score: {hit.score:.2f}]")
                print(f"{hit.payload.get('_text', '')[:200]}...")
                
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
