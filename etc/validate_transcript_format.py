#!/usr/bin/env python
"""
Validates that a transcript file matches the expected format for the dashboard.

Expected format (from server.py and dashboard):
    speaker: HH:MM:SS.fff text

Examples:
    S1: 00:00:02.350 Hello, how are you?
    S2: 00:00:06.000 I'm doing well, thanks!
    <speaker:1>: 00:02:22.440 The text goes here

Usage:
    python validate_transcript_format.py <transcript_file_path>
"""

import sys
from pathlib import Path
import re

# Setup path
file = Path(__file__).resolve()
ROOT = None
for parent in file.parents:
    if parent.name == "dpinterview":
        ROOT = parent
        break
sys.path.append(str(ROOT))

import argparse
import logging
from rich.logging import RichHandler
from rich.console import Console
from rich.table import Table
import pandas as pd

MODULE_NAME = "validate_transcript_format"
logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = Console()


def validate_transcript(transcript_path: Path) -> bool:
    """
    Validate that a transcript file matches the expected format.

    Expected format:
        speaker: HH:MM:SS.fff text

    Where:
        - speaker can be "S1", "S2", "<speaker:1>", etc.
        - HH:MM:SS.fff is a timestamp with milliseconds
        - text is the transcript text

    Args:
        transcript_path (Path): Path to the transcript file

    Returns:
        bool: True if valid, False otherwise
    """
    if not transcript_path.exists():
        logger.error(f"File not found: {transcript_path}")
        return False

    logger.info(f"Validating transcript: {transcript_path}")
    logger.info(f"File size: {transcript_path.stat().st_size / 1024:.2f} KB")

    # Read file
    with open(transcript_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Total lines: {len(lines)}")

    # Validation patterns
    # Dashboard expects: S\d+: HH:MM:SS.fff text
    dashboard_pattern = re.compile(r'^(S\d+):\s*([\d:.]+)\s+(.+)')

    # Server.py expects: speaker: HH:MM:SS.fff text (with colon after speaker)
    server_pattern = re.compile(r'^(.+?):\s*([\d:]+\.[\d]+)\s+(.+)')

    valid_lines = []
    invalid_lines = []
    empty_lines = 0

    for i, line in enumerate(lines, start=1):
        stripped = line.strip()

        if not stripped:
            empty_lines += 1
            continue

        # Check against dashboard pattern
        dashboard_match = dashboard_pattern.match(stripped)
        # Check against server pattern
        server_match = server_pattern.match(stripped)

        if dashboard_match or server_match:
            match = dashboard_match or server_match
            speaker = match.group(1)
            time = match.group(2)
            text = match.group(3)

            # Validate timestamp format
            try:
                pd.to_datetime(time, format="%H:%M:%S.%f")
                valid_lines.append({
                    "line": i,
                    "speaker": speaker,
                    "time": time,
                    "text": text[:50] + "..." if len(text) > 50 else text
                })
            except ValueError:
                invalid_lines.append({
                    "line": i,
                    "content": stripped[:80] + "..." if len(stripped) > 80 else stripped,
                    "reason": f"Invalid timestamp format: {time}"
                })
        else:
            invalid_lines.append({
                "line": i,
                "content": stripped[:80] + "..." if len(stripped) > 80 else stripped,
                "reason": "Does not match expected format"
            })

    # Display results
    console.rule("[bold blue]Validation Results[/bold blue]")

    total_lines = len(lines)
    valid_count = len(valid_lines)
    invalid_count = len(invalid_lines)

    logger.info(f"Total lines: {total_lines}")
    logger.info(f"Empty lines: {empty_lines}")
    logger.info(f"[green]Valid lines: {valid_count}[/green]", extra={"markup": True})
    logger.info(f"[red]Invalid lines: {invalid_count}[/red]", extra={"markup": True})

    # Show sample valid lines
    if valid_lines:
        console.rule("[bold green]Sample Valid Lines (first 5)[/bold green]")
        table = Table(show_header=True)
        table.add_column("Line #", style="dim")
        table.add_column("Speaker", style="cyan")
        table.add_column("Time", style="magenta")
        table.add_column("Text", style="white")

        for line in valid_lines[:5]:
            table.add_row(
                str(line["line"]),
                line["speaker"],
                line["time"],
                line["text"]
            )

        console.print(table)

    # Show invalid lines
    if invalid_lines:
        console.rule("[bold red]Invalid Lines[/bold red]")
        table = Table(show_header=True)
        table.add_column("Line #", style="dim")
        table.add_column("Content", style="white")
        table.add_column("Reason", style="red")

        for line in invalid_lines[:10]:  # Show first 10
            table.add_row(
                str(line["line"]),
                line["content"],
                line["reason"]
            )

        if len(invalid_lines) > 10:
            logger.warning(f"... and {len(invalid_lines) - 10} more invalid lines")

        console.print(table)

    # Final verdict
    console.rule("[bold blue]Verdict[/bold blue]")

    if invalid_count == 0:
        logger.info("[bold green]✓ Transcript is valid and ready for import![/bold green]", extra={"markup": True})
        return True
    else:
        percentage_valid = (valid_count / total_lines) * 100
        logger.warning(
            f"[bold yellow]⚠ Transcript has {invalid_count} invalid lines ({percentage_valid:.1f}% valid)[/bold yellow]",
            extra={"markup": True}
        )
        logger.info("The transcript may still work if most lines are valid.")
        logger.info("Invalid lines will be skipped during parsing.")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Validate transcript file format for dpinterview dashboard.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Expected format:
    speaker: HH:MM:SS.fff text

Examples of valid lines:
    S1: 00:00:02.350 Hello, how are you?
    S2: 00:00:06.000 I'm doing well, thanks!
    <speaker:1>: 00:02:22.440 The text goes here

Usage example:
    python validate_transcript_format.py /path/to/transcript.txt
        """
    )
    parser.add_argument(
        "transcript_path",
        type=str,
        help="Path to the transcript file"
    )

    args = parser.parse_args()
    transcript_path = Path(args.transcript_path).resolve()

    is_valid = validate_transcript(transcript_path)
    sys.exit(0 if is_valid else 1)
