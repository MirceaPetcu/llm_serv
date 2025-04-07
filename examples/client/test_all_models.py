"""
This script tests all available models on a running LLM service server.
It sends a simple "1+1=" query to each model and reports on which models
worked successfully and which failed, along with timing information.

To use this script:
1. Ensure the LLM service server is running
2. Run this script with: python -m examples.client.test_all_models
"""

import asyncio
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from llm_serv import LLMServiceClient, Conversation, LLMRequest, LLMResponse
from llm_serv.exceptions import ServiceCallException, TimeoutException

console = Console()

@dataclass
class ModelTestResult:
    provider: str
    model: str
    success: bool
    response: Optional[LLMResponse] = None
    error_message: str = ""
    time_taken: float = 0.0

async def test_model(client: LLMServiceClient, provider: str, model_name: str, timeout: float = 30.0) -> ModelTestResult:
    """Test a single model with a simple query."""
    start_time = time.time()
    result = ModelTestResult(provider=provider, model=model_name, success=False)
    
    try:
        client.set_model(provider=provider, name=model_name)
        conversation = Conversation.from_prompt("1+1=")
        request = LLMRequest(
            conversation=conversation,
            max_completion_tokens=10,
            temperature=0.0
        )
        
        # Make the API call with timeout
        response = await client.chat(request, timeout=timeout)
        
        # If we get here, test was successful
        result.success = True
        result.response = response
        
    except TimeoutException as e:
        result.error_message = f"Timeout after {timeout} seconds"
    except ServiceCallException as e:
        result.error_message = str(e)
    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)}"
    
    result.time_taken = time.time() - start_time
    return result

async def main():
    # Initialize client
    client = LLMServiceClient(host="localhost", port=9999, timeout=5.0)
    
    # Check server health
    try:
        await client.server_health_check(timeout=2.0)
        console.print("\n[bold green]Server health check: OK[/bold green]")
    except Exception as e:
        console.print(f"\n[bold red]Server health check failed: {e}[/bold red]")
        console.print("[yellow]Make sure the server is running (python -m llm_serv.server)[/yellow]")
        return
    
    # Get available models
    try:
        all_models = await client.list_models()
        console.print(f"\nFound [bold]{len(all_models)}[/bold] models across all providers")
    except Exception as e:
        console.print(f"\n[bold red]Failed to retrieve models: {e}[/bold red]")
        return
    
    # Test each model
    results: List[ModelTestResult] = []
    
    with console.status("[bold green]Testing models...[/bold green]") as status:
        for model_info in all_models:
            provider = model_info["provider"]["name"]
            model_name = model_info["name"]
            
            status.update(f"[bold green]Testing {provider}/{model_name}...[/bold green]")
            result = await test_model(client, provider, model_name)
            results.append(result)
            
            # Print immediate result
            if result.success:
                console.print(f"[green]✓[/green] {provider}/{model_name}: {result.response.output.strip()} ({result.time_taken:.2f}s)")
            else:
                console.print(f"[red]✗[/red] {provider}/{model_name}: {result.error_message} ({result.time_taken:.2f}s)")
    
    # Generate final report
    console.print("\n[bold]Test Summary[/bold]")
    
    # Group by provider using string keys
    by_provider = defaultdict(list)
    for result in results:
        # Ensure provider is a string for use as a dictionary key
        provider_key = str(result.provider)
        by_provider[provider_key].append(result)
    
    # Create a table for the summary
    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Response")
    table.add_column("Time (s)")
    
    for provider, provider_results in by_provider.items():
        # Sort by model name
        provider_results.sort(key=lambda x: x.model)
        
        for i, result in enumerate(provider_results):
            # Only show provider name for first model of each provider
            provider_display = provider if i == 0 else ""
            
            status = Text("✓", style="green") if result.success else Text("✗", style="red")
            
            # Safe handling of response text
            if result.success and hasattr(result.response, 'output'):
                response_text = str(result.response.output).strip()
            else:
                response_text = result.error_message
                
            if len(response_text) > 30:
                response_text = response_text[:27] + "..."
                
            table.add_row(
                provider_display, 
                result.model,
                status,
                response_text,
                f"{result.time_taken:.2f}"
            )
    
    console.print(table)
    
    # Print statistics
    successful = sum(1 for r in results if r.success)
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"Total models tested: {len(results)}")
    console.print(f"Successful: [green]{successful}[/green]")
    console.print(f"Failed: [red]{len(results) - successful}[/red]")
    
    # Show the fastest and slowest successful models
    successful_results = [r for r in results if r.success]
    if successful_results:
        fastest = min(successful_results, key=lambda x: x.time_taken)
        slowest = max(successful_results, key=lambda x: x.time_taken)
        
        console.print(f"\nFastest model: [green]{fastest.provider}/{fastest.model}[/green] ({fastest.time_taken:.2f}s)")
        console.print(f"Slowest model: [yellow]{slowest.provider}/{slowest.model}[/yellow] ({slowest.time_taken:.2f}s)")

if __name__ == "__main__":
    asyncio.run(main())
