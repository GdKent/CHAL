"""
main.py

CLI entry point for CHAL — banner, argument parsing, mode routing.

Usage:
    chal                              # Interactive wizard mode
    chal --config path/to.yaml        # Headless mode
    chal --config path/to.yaml --edit # Load config into wizard for editing
    chal --history                    # Show past debates
    chal --replay <id>                # Re-run a past debate
"""

import argparse
import sys

from rich.console import Console
from rich.text import Text

from chal.cli.runner import run_debate
from chal.cli.wizard import run_wizard
from chal.config import load_config

CHAL_BANNER = r"""
 ██████╗██╗  ██╗ █████╗ ██╗
██╔════╝██║  ██║██╔══██╗██║
██║     ███████║███████║██║
██║     ██╔══██║██╔══██║██║
╚██████╗██║  ██║██║  ██║███████╗
 ╚═════╝╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝
"""

CHAL_TAGLINE = "Council of Hierarchical Agentic Language"


def get_version() -> str:
    """Get package version from metadata."""
    try:
        from importlib.metadata import version
        return version("chal")
    except Exception:
        return "dev"


def show_banner(console: Console) -> None:
    """Display the CHAL ASCII banner with version info."""
    banner_text = Text(CHAL_BANNER, style="bold #9B1B30")
    console.print(banner_text, highlight=False)

    version = get_version()
    console.print(
        f"  [#C75B7A]{CHAL_TAGLINE}[/#C75B7A]  [dim]v{version}[/dim]\n",
        highlight=False,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments.

    Args:
        argv: Argument list (defaults to sys.argv[1:] if None).
    """
    parser = argparse.ArgumentParser(
        prog="chal",
        description="CHAL — Interactive multi-agent philosophical debate framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  chal                              Interactive wizard
  chal --config default             Run with built-in default config
  chal --config my_debate.yaml      Run with custom YAML file
  chal --config default --edit      Load config into wizard for editing
  chal --history                    Show past debates
  chal --replay a1b2c3d4            Re-run a past debate by ID
""",
    )
    parser.add_argument(
        "--config", "-c",
        default=None,
        help="Configuration name or YAML file path (headless mode)",
    )
    parser.add_argument(
        "--edit", "-e",
        action="store_true",
        help="Load config into the wizard for editing before launch",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--history",
        action="store_true",
        help="Display past debate history",
    )
    parser.add_argument(
        "--replay",
        default=None,
        metavar="ID",
        help="Re-run a past debate by its history ID",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """CHAL CLI entry point."""
    console = Console()
    show_banner(console)

    args = parse_args(argv)

    # --history: show debate history and exit
    if args.history:
        from chal.cli.history import list_debates, format_history_table
        debates = list_debates()
        format_history_table(debates, console)
        return 0

    # --replay <id>: reload a past debate config and run it
    if args.replay:
        from chal.cli.history import load_debate_config
        try:
            config = load_debate_config(args.replay)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
        return run_debate(config, console, verbose=args.verbose, interactive=False)

    # --edit without --config makes no sense
    if args.edit and not args.config:
        console.print("[red]--edit requires --config to specify a config to edit[/red]")
        return 1

    if args.config and not args.edit:
        # Headless mode: load config and run directly
        try:
            config = load_config(args.config)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
        except Exception as e:
            console.print(f"[red]Error loading configuration:[/red] {e}")
            return 1

        return run_debate(config, console, verbose=args.verbose, interactive=False)

    elif args.config and args.edit:
        # Load config, then open wizard with pre-filled values
        try:
            prefill = load_config(args.config)
        except FileNotFoundError as e:
            console.print(f"[red]Error:[/red] {e}")
            return 1
        except Exception as e:
            console.print(f"[red]Error loading configuration:[/red] {e}")
            return 1

        config, action = run_wizard(console, prefill=prefill)

        if action == "cancel" or config is None:
            console.print("[dim]Cancelled.[/dim]")
            return 0

        if action == "launch":
            return run_debate(config, console, verbose=args.verbose)

        return 0

    else:
        # Interactive wizard mode (no args)
        try:
            config, action = run_wizard(console)
        except KeyboardInterrupt:
            console.print("\n[dim]Cancelled.[/dim]")
            return 0

        if action == "cancel" or config is None:
            console.print("[dim]Cancelled.[/dim]")
            return 0

        if action == "launch":
            return run_debate(config, console, verbose=args.verbose)

        return 0


if __name__ == "__main__":
    sys.exit(main())
