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
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text

from llm_serv import Conversation, LLMRequest, LLMResponse, LLMServiceClient
from llm_serv.core.exceptions import ServiceCallException, TimeoutException

console = Console()

@dataclass
class ModelTestResult:
    model_id: str  # Full model identifier
    provider: str  # Extracted provider name
    model_name: str  # Extracted model name
    success: bool
    response: Optional[LLMResponse] = None
    error_message: str = ""
    time_taken: float = 0.0

def parse_model_id(model_id: str) -> tuple[str, str]:
    """Parse the model ID to extract provider and model name"""
    # Example: name='AWS' config={}/claude-3-5-sonnet
    match = re.match(r"name='([^']+)'[^/]+/(.+)$", model_id)
    if match:
        provider = match.group(1)
        model_name = match.group(2)
        return provider, model_name
    return "Unknown", model_id

async def test_model(client: LLMServiceClient, model_id: str, timeout: float = 30.0) -> ModelTestResult:
    """Test a single model with a simple query."""
    start_time = time.time()
    provider, model_name = parse_model_id(model_id)
    result = ModelTestResult(
        model_id=model_id,
        provider=provider,
        model_name=model_name,
        success=False
    )
    
    try:
        # Set the model using the full model_id as provided by the server
        client.set_model(model_id)
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
        for model_id in all_models:
            status.update(f"[bold green]Testing {model_id}...[/bold green]")
            result = await test_model(client, model_id)
            results.append(result)
            
            # Print immediate result
            if result.success:
                console.print(f"[green]✓[/green] {model_id}: {result.response.output.strip()} ({result.time_taken:.2f}s)")
            else:
                console.print(f"[red]✗[/red] {model_id}: {result.error_message} ({result.time_taken:.2f}s)")
    
    # Generate final report
    console.print("\n[bold]Test Summary[/bold]")
    
    # Group by provider using the extracted provider name
    by_provider = defaultdict(list)
    for result in results:
        by_provider[result.provider].append(result)
    
    # Create a table for the summary
    table = Table(show_header=True, header_style="bold")
    table.add_column("Provider")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Response")
    table.add_column("Time (s)")
    
    for provider, provider_results in by_provider.items():
        # Sort by model name
        provider_results.sort(key=lambda x: x.model_name)
        
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
                result.model_name,
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
        
        console.print(f"\nFastest model: [green]{fastest.provider}/{fastest.model_name}[/green] ({fastest.time_taken:.2f}s)")
        console.print(f"Slowest model: [yellow]{slowest.provider}/{slowest.model_name}[/yellow] ({slowest.time_taken:.2f}s)")

if __name__ == "__main__":
    asyncio.run(main())
