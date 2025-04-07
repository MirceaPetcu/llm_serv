"""
This script tests concurrent requests to the LLM service.
It sends 10 concurrent queries to the OpenAI GPT-4o-mini model
and measures the timing performance.

To use this script:
1. Ensure the LLM service server is running
2. Run this script with: python -m examples.client.concurrency_test
"""

import asyncio
import time
from dataclasses import dataclass
from typing import List, Optional

from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from llm_serv.client import LLMServiceClient
from llm_serv.conversation import Conversation
from llm_serv.exceptions import ServiceCallException, TimeoutException
from llm_serv.providers.base import LLMRequest, LLMResponse

console = Console()

@dataclass
class ConcurrencyTestResult:
    """Store the results of a concurrent test execution"""
    query_id: int
    success: bool
    response: Optional[LLMResponse] = None
    error_message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    
    @property
    def time_taken(self) -> float:
        return self.end_time - self.start_time

async def run_query(client: LLMServiceClient, query_id: int, timeout: float = 60.0) -> ConcurrencyTestResult:
    """Run a single query and record the results"""
    result = ConcurrencyTestResult(query_id=query_id, success=False)
    result.start_time = time.time()
    
    # Log when request is being sent
    console.print(f"[cyan]Query {query_id}: Starting request at {time.strftime('%H:%M:%S.%f')[:-3]}[/cyan]")
    
    try:
        # Create the conversation with our simple prompt
        conversation = Conversation.from_prompt("Write 10 names.")
        request = LLMRequest(
            conversation=conversation,
            max_completion_tokens=100,
            temperature=0.7
        )
        
        # Make the API call with timeout
        response = await client.chat(request, timeout=timeout)
        
        # Test was successful
        result.success = True
        result.response = response
        console.print(f"[green]Query {query_id}: Received response at {time.strftime('%H:%M:%S.%f')[:-3]}[/green]")
        
    except TimeoutException as e:
        result.error_message = f"Timeout after {timeout} seconds"
        console.print(f"[red]Query {query_id}: Timeout after {timeout} seconds[/red]")
    except ServiceCallException as e:
        result.error_message = str(e)
        console.print(f"[red]Query {query_id}: Service error - {str(e)}[/red]")
    except Exception as e:
        result.error_message = f"Unexpected error: {str(e)}"
        console.print(f"[red]Query {query_id}: Unexpected error - {str(e)}[/red]")
    
    result.end_time = time.time()
    return result

async def main():
    # Configure parameters
    NUM_CONCURRENT_QUERIES = 10
    TIMEOUT = 60.0  # seconds
    
    console.print(f"\n[bold]Setting up {NUM_CONCURRENT_QUERIES} concurrent requests...[/bold]")
    
    # Create tasks with separate clients
    query_tasks = []
    for i in range(NUM_CONCURRENT_QUERIES):
        client = LLMServiceClient(host="localhost", port=9999, timeout=TIMEOUT)
        #client.set_model(provider="OPENAI", name="gpt-4o-mini")
        client.set_model(provider="AWS", name="claude-3-haiku")
        query_tasks.append(run_query(client, i+1, TIMEOUT))
    
    # Run all queries concurrently with a progress indicator
    console.print(f"\n[bold]Running {NUM_CONCURRENT_QUERIES} concurrent queries at {time.strftime('%H:%M:%S.%f')[:-3]}...[/bold]")
    
    # Track when all requests are actually sent
    start_all = time.time()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        task = progress.add_task(f"Processing {NUM_CONCURRENT_QUERIES} concurrent requests...", total=1)
        results = await asyncio.gather(*query_tasks)
        progress.update(task, completed=1)
    
    end_all = time.time()
    total_wall_time = end_all - start_all
    console.print(f"\n[bold]All requests completed in {total_wall_time:.2f} seconds[/bold]")
    
    # Calculate statistics and show detailed timing information
    successful = sum(1 for r in results if r.success)
    failed = NUM_CONCURRENT_QUERIES - successful
    
    if successful > 0:
        avg_time = sum(r.time_taken for r in results if r.success) / successful
        min_time = min((r.time_taken for r in results if r.success), default=0)
        max_time = max((r.time_taken for r in results if r.success), default=0)
        total_time = sum(r.time_taken for r in results if r.success)
        
        # Compare the sum of individual times with the wall-clock time
        console.print(f"Sum of individual request times: {total_time:.2f}s")
        console.print(f"Wall-clock time for all requests: {total_wall_time:.2f}s")
        console.print(f"Concurrency factor: {total_time/total_wall_time:.2f}x")
    else:
        avg_time = min_time = max_time = 0
    
    # Create a table for the results
    table = Table(show_header=True, header_style="bold")
    table.add_column("Query ID")
    table.add_column("Status")
    table.add_column("Time (s)")
    table.add_column("Names Generated")
    
    # Sort results by query_id
    results.sort(key=lambda x: x.query_id)
    
    for result in results:
        status = Text("✓", style="green") if result.success else Text("✗", style="red")
        
        time_taken = f"{result.time_taken:.2f}"
        
        if result.success:
            # Extract a snippet of the names from response output
            response_text = result.response.output.strip()
            # Try to format the output for better display
            names = response_text.replace("\n", ", ")
            if len(names) > 60:
                names = names[:57] + "..."
        else:
            names = result.error_message
            
        table.add_row(
            str(result.query_id),
            status,
            time_taken,
            names
        )
    
    # Display results
    console.print("\n[bold]Test Results:[/bold]")
    console.print(table)
    
    # Print statistics
    console.print(f"\n[bold]Statistics:[/bold]")
    console.print(f"Total queries: {NUM_CONCURRENT_QUERIES}")
    console.print(f"Successful: [green]{successful}[/green]")
    console.print(f"Failed: [red]{failed}[/red]")
    console.print(f"Average time: [cyan]{avg_time:.2f}s[/cyan]")
    console.print(f"Minimum time: [green]{min_time:.2f}s[/green]")
    console.print(f"Maximum time: [yellow]{max_time:.2f}s[/yellow]")
    
    # Calculate concurrency efficiency
    if successful > 0 and max_time > 0:
        # If perfectly concurrent, total time would equal max time
        # efficiency = (sum of individual times) / (max time * number of successful queries)
        perfect_concurrent_time = max_time
        sequential_time = sum(r.time_taken for r in results if r.success)
        efficiency = sequential_time / (perfect_concurrent_time * successful) if successful > 0 else 0
        console.print(f"Concurrency efficiency: [magenta]{efficiency:.2f}x[/magenta] (higher is better)")

if __name__ == "__main__":
    asyncio.run(main())
