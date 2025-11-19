#!/usr/bin/env python3
"""
QuantCLI AI Copilot CLI

Interactive AI assistant for trading system analysis and insights.

Usage:
    python scripts/copilot.py ask "What's my portfolio status?"
    python scripts/copilot.py analyze-signal AAPL
    python scripts/copilot.py explain-portfolio
    python scripts/copilot.py market-insight TSLA
    python scripts/copilot.py chat  # Interactive mode
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import click
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from loguru import logger

from src.copilot.service import CopilotService
from src.copilot.context import ContextProvider
from src.copilot.explainer import ModelExplainer

# Initialize rich console for beautiful output
console = Console()


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(verbose):
    """
    QuantCLI AI Copilot - Your intelligent trading assistant.

    Powered by open-source LLMs for context-aware trading insights.
    """
    if verbose:
        logger.add(sys.stderr, level="DEBUG")
    else:
        logger.remove()
        logger.add(sys.stderr, level="INFO")


@cli.command()
@click.argument('question')
@click.option('--symbol', '-s', help='Focus on specific symbol')
@click.option('--context', '-c', is_flag=True, help='Include full trading context')
def ask(question: str, symbol: str, context: bool):
    """
    Ask the copilot a question about your trading system.

    Examples:
        copilot ask "What's my current portfolio status?"
        copilot ask "Should I buy AAPL?" --symbol AAPL
        copilot ask "How is my strategy performing?" --context
    """
    console.print(Panel.fit(
        f"[bold cyan]Question:[/bold cyan] {question}",
        title="AI Copilot Query"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Thinking...", total=None)

        # Get context if requested
        context_data = {}
        if context or symbol:
            progress.update(task, description="Gathering context...")
            context_provider = ContextProvider()
            context_data = context_provider.get_full_context(symbol=symbol)

        # Get copilot response
        progress.update(task, description="Generating response...")
        copilot = CopilotService()
        response = copilot.ask(question, context=context_data)

        progress.update(task, description="Done!", completed=True)

    # Display response
    console.print()
    console.print(Panel(
        Markdown(response),
        title="[bold green]Copilot Response[/bold green]",
        border_style="green"
    ))


@cli.command()
@click.argument('symbol')
@click.option('--hours', '-h', default=24, help='Hours of signal history to analyze')
def analyze_signal(symbol: str, hours: int):
    """
    Analyze recent trading signals for a symbol.

    Example:
        copilot analyze-signal AAPL --hours 48
    """
    console.print(Panel.fit(
        f"[bold cyan]Analyzing signals for {symbol}[/bold cyan]",
        title="Signal Analysis"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading data...", total=None)

        # Get context
        context_provider = ContextProvider()

        progress.update(task, description="Fetching recent signals...")
        signals = context_provider.get_recent_signals(symbol=symbol, hours=hours, limit=10)

        progress.update(task, description="Fetching market data...")
        market_data = context_provider.get_market_data(symbol=symbol, days=7)

        # Get recent features (placeholder - would come from feature store)
        features = {"note": "Feature data integration pending"}

        if not signals:
            console.print("[yellow]No recent signals found for this symbol[/yellow]")
            return

        # Analyze with copilot
        progress.update(task, description="Analyzing with AI...")
        copilot = CopilotService()

        signal_data = {
            "symbol": symbol,
            "recent_signals": signals[:3],  # Top 3
            "total_signals": len(signals),
        }

        response = copilot.analyze_signal(
            signal_data=signal_data,
            market_context=market_data,
            features=features,
        )

        progress.update(task, description="Done!", completed=True)

    # Display results
    console.print()
    console.print(Panel(
        Markdown(response),
        title=f"[bold green]Signal Analysis: {symbol}[/bold green]",
        border_style="green"
    ))


@cli.command()
@click.option('--days', '-d', default=30, help='Days of performance data to analyze')
def explain_portfolio(days: int):
    """
    Get AI analysis of your current portfolio.

    Example:
        copilot explain-portfolio --days 30
    """
    console.print(Panel.fit(
        "[bold cyan]Analyzing your portfolio[/bold cyan]",
        title="Portfolio Analysis"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading portfolio data...", total=None)

        # Get context
        context_provider = ContextProvider()

        progress.update(task, description="Fetching portfolio summary...")
        portfolio = context_provider.get_portfolio_summary()

        progress.update(task, description="Fetching positions...")
        positions = context_provider.get_current_positions(limit=20)

        progress.update(task, description="Calculating performance metrics...")
        performance = context_provider.get_performance_metrics(days=days)

        market_context = {"period": f"{days} days"}

        # Analyze with copilot
        progress.update(task, description="Generating AI insights...")
        copilot = CopilotService()

        response = copilot.analyze_portfolio(
            portfolio_data=portfolio,
            positions=positions,
            performance_metrics=performance,
            market_context=market_context,
        )

        progress.update(task, description="Done!", completed=True)

    # Display portfolio summary table
    console.print()
    table = Table(title="Portfolio Summary", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Total Value", f"${portfolio.get('total_value', 0):,.2f}")
    table.add_row("Unrealized P&L", f"${portfolio.get('total_unrealized_pnl', 0):,.2f}")
    table.add_row("Number of Positions", str(portfolio.get('num_positions', 0)))
    table.add_row("Total Trades", str(performance.get('num_trades', 0)))
    table.add_row("Win Rate", f"{performance.get('win_rate', 0):.1f}%")
    table.add_row("Avg P&L per Trade", f"${performance.get('avg_pnl', 0):,.2f}")

    console.print(table)
    console.print()

    # Display AI analysis
    console.print(Panel(
        Markdown(response),
        title="[bold green]AI Portfolio Analysis[/bold green]",
        border_style="green"
    ))


@cli.command()
@click.argument('symbol')
@click.option('--days', '-d', default=30, help='Days of market data to analyze')
def market_insight(symbol: str, days: int):
    """
    Get AI insights on market conditions for a symbol.

    Example:
        copilot market-insight AAPL --days 30
    """
    console.print(Panel.fit(
        f"[bold cyan]Analyzing market conditions for {symbol}[/bold cyan]",
        title="Market Insight"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading market data...", total=None)

        # Get context
        context_provider = ContextProvider()

        progress.update(task, description="Fetching market data...")
        market_data = context_provider.get_market_data(symbol=symbol, days=days)

        # Placeholder for indicators and sentiment
        indicators = {"note": "Technical indicators integration pending"}
        news_sentiment = {"note": "News sentiment integration pending"}

        # Generate insights
        progress.update(task, description="Generating AI insights...")
        copilot = CopilotService()

        response = copilot.interpret_market(
            market_data=market_data,
            indicators=indicators,
            news_sentiment=news_sentiment,
            historical_context=f"Last {days} days",
        )

        progress.update(task, description="Done!", completed=True)

    # Display results
    console.print()
    console.print(Panel(
        Markdown(response),
        title=f"[bold green]Market Insight: {symbol}[/bold green]",
        border_style="green"
    ))


@cli.command()
def chat():
    """
    Start an interactive chat session with the AI copilot.

    Type 'exit' or 'quit' to end the session.
    """
    console.print(Panel.fit(
        "[bold green]QuantCLI AI Copilot - Interactive Mode[/bold green]\n"
        "Ask me anything about your trading system!\n"
        "Commands: exit, quit, clear, help",
        title="AI Copilot Chat"
    ))

    copilot = CopilotService()
    context_provider = ContextProvider()

    console.print()

    while True:
        try:
            # Get user input
            question = console.input("[bold cyan]You:[/bold cyan] ")

            if not question.strip():
                continue

            # Handle special commands
            if question.lower() in ['exit', 'quit']:
                console.print("[yellow]Goodbye![/yellow]")
                break

            if question.lower() == 'clear':
                console.clear()
                continue

            if question.lower() == 'help':
                console.print("""
[bold]Available Commands:[/bold]
- Ask any question about your portfolio, trades, signals, or strategy
- 'exit' or 'quit' - End the chat session
- 'clear' - Clear the screen
- 'help' - Show this help message

[bold]Example Questions:[/bold]
- What's my current portfolio status?
- Why did I get a buy signal for AAPL?
- How is my strategy performing?
- What are the top features in my model?
- Should I adjust my risk parameters?
                """)
                continue

            # Get context
            with console.status("[bold green]Thinking..."):
                context_data = context_provider.get_full_context()
                response = copilot.ask(question, context=context_data)

            # Display response
            console.print()
            console.print(Panel(
                Markdown(response),
                title="[bold green]Copilot[/bold green]",
                border_style="green"
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[yellow]Goodbye![/yellow]")
            break
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")


@cli.command()
@click.option('--model', '-m', help='Specific model name to load')
@click.option('--cpu', is_flag=True, help='Force CPU inference')
def load_model(model: str, cpu: bool):
    """
    Pre-load the AI model for faster responses.

    Example:
        copilot load-model
        copilot load-model --model microsoft/Phi-3-mini-4k-instruct
    """
    console.print(Panel.fit(
        "[bold cyan]Loading AI model...[/bold cyan]",
        title="Model Loading"
    ))

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading...", total=None)

        try:
            copilot = CopilotService()
            copilot.load_model(model_name=model, force_cpu=cpu)

            progress.update(task, description="Done!", completed=True)
            console.print()
            console.print("[bold green]✓[/bold green] Model loaded successfully!")

        except Exception as e:
            console.print(f"[red]Error loading model: {e}[/red]")
            sys.exit(1)


@cli.command()
def clear_cache():
    """
    Clear the copilot response cache.
    """
    console.print("[yellow]Clearing cache...[/yellow]")

    copilot = CopilotService()
    copilot.clear_cache()

    console.print("[bold green]✓[/bold green] Cache cleared!")


@cli.command()
def status():
    """
    Show copilot status and configuration.
    """
    copilot = CopilotService()
    context_provider = ContextProvider()

    # Create status table
    table = Table(title="Copilot Status", show_header=True, header_style="bold magenta")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")

    # Check model status
    model_status = "Loaded ✓" if copilot.model is not None else "Not loaded"
    table.add_row("AI Model", model_status)

    # Check cache
    cache_size = len(copilot._cache)
    table.add_row("Cache Size", f"{cache_size} responses")

    # Check database
    try:
        portfolio = context_provider.get_portfolio_summary()
        db_status = "Connected ✓"
    except Exception:
        db_status = "Not connected"

    table.add_row("Database", db_status)

    console.print(table)

    # Model info
    if copilot.model is not None:
        console.print()
        console.print(Panel(
            f"[bold]Model:[/bold] {copilot._get_model_name()}\n"
            f"[bold]Device:[/bold] {'GPU' if copilot.model.device.type == 'cuda' else 'CPU'}",
            title="Model Information",
            border_style="blue"
        ))


if __name__ == "__main__":
    cli()
