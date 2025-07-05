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
    model_id: str
    success: bool
    response: Optional[LLMResponse] = None
    error_message: str = ""
    time_taken: float = 0.0

async def test_model(client: LLMServiceClient, model_id: str, timeout: float = 30.0) -> ModelTestResult:
    """Test a single model with a simple query."""
    start_time = time.time()
    result = ModelTestResult(
        model_id=model_id,
        success=False
    )
    
    try:
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
    
    # Sort results by model_id
    results.sort(key=lambda x: x.model_id)
    
    # Create a table for the summary
    table = Table(show_header=True, header_style="bold")
    table.add_column("Model")
    table.add_column("Status")
    table.add_column("Response")
    table.add_column("Time (s)")
    
    for result in results:
        status = Text("✓", style="green") if result.success else Text("✗", style="red")
        
        # Safe handling of response text
        if result.success and hasattr(result.response, 'output'):
            response_text = str(result.response.output).strip()
        else:
            response_text = result.error_message
            
        if len(response_text) > 30:
            response_text = response_text[:27] + "..."
            
        table.add_row(
            result.model_id,
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
        
        console.print(f"\nFastest model: [green]{fastest.model_id}[/green] ({fastest.time_taken:.2f}s)")
        console.print(f"Slowest model: [yellow]{slowest.model_id}[/yellow] ({slowest.time_taken:.2f}s)")

if __name__ == "__main__":
    asyncio.run(main())
