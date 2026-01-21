#!/usr/bin/env python
"""
Check if an interview exists in the database and show its details.

This helps debug transcript import issues by verifying the interview
is properly set up in the database.
"""

import sys
from pathlib import Path

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

from pipeline.helpers import utils, db

MODULE_NAME = "check_interview_exists"
logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = Console()


def check_interview(interview_name: str, config_file: Path) -> None:
    """
    Check if an interview exists in the database.
    """

    console.rule(f"[bold blue]Checking Interview: {interview_name}[/bold blue]")

    # Check interviews table
    query = f"""
        SELECT * FROM interviews
        WHERE interview_name = '{interview_name}';
    """

    interviews_df = db.execute_sql(config_file=config_file, query=query)

    if interviews_df.empty:
        logger.error(f"[red]✗ Interview '{interview_name}' NOT FOUND in 'interviews' table[/red]", extra={"markup": True})
        logger.info("\nThis interview doesn't exist in the database yet.")
        logger.info("You need to run the appropriate crawler to import this interview first.")
        return
    else:
        logger.info(f"[green]✓ Interview found in 'interviews' table[/green]", extra={"markup": True})
        interview_path = interviews_df.iloc[0]['interview_path']
        study_id = interviews_df.iloc[0]['study_id']
        logger.info(f"  Interview path: {interview_path}")
        logger.info(f"  Study ID: {study_id}")

    # Check interview_parts table
    query = f"""
        SELECT * FROM interview_parts
        WHERE interview_path = '{interview_path}';
    """

    parts_df = db.execute_sql(config_file=config_file, query=query)

    if parts_df.empty:
        logger.warning(f"[yellow]⚠ No entries in 'interview_parts' table[/yellow]", extra={"markup": True})
    else:
        logger.info(f"[green]✓ Found {len(parts_df)} parts in 'interview_parts' table[/green]", extra={"markup": True})

    # Check interview_files table
    query = f"""
        SELECT * FROM interview_files
        WHERE interview_path = '{interview_path}';
    """

    files_df = db.execute_sql(config_file=config_file, query=query)

    if files_df.empty:
        logger.warning(f"[yellow]⚠ No files in 'interview_files' table[/yellow]", extra={"markup": True})
    else:
        logger.info(f"[green]✓ Found {len(files_df)} files in 'interview_files' table[/green]", extra={"markup": True})

        # Show files with transcript tag
        transcript_files = files_df[files_df['interview_file_tags'].str.contains('transcript', na=False)]
        if not transcript_files.empty:
            logger.info(f"  - {len(transcript_files)} transcript file(s)")

    # Check transcript_files table
    query = f"""
        SELECT * FROM transcript_files
        WHERE identifier_name = '{interview_name}';
    """

    transcript_files_df = db.execute_sql(config_file=config_file, query=query)

    if transcript_files_df.empty:
        logger.warning(f"[yellow]⚠ No entries in 'transcript_files' table[/yellow]", extra={"markup": True})
    else:
        logger.info(f"[green]✓ Found {len(transcript_files_df)} entries in 'transcript_files' table[/green]", extra={"markup": True})
        for _, row in transcript_files_df.iterrows():
            transcript_file = row['transcript_file']
            tags = row['transcript_file_tags']
            logger.info(f"  - {transcript_file}")
            logger.info(f"    Tags: {tags}")

            # Check if file exists
            if Path(transcript_file).exists():
                logger.info(f"    [green]✓ File exists on disk[/green]", extra={"markup": True})
            else:
                logger.error(f"    [red]✗ File NOT found on disk[/red]", extra={"markup": True})

    # Summary
    console.rule("[bold blue]Summary[/bold blue]")

    can_use_dashboard = not interviews_df.empty and not transcript_files_df.empty
    can_run_qc = not interviews_df.empty and not files_df[files_df['interview_file_tags'].str.contains('transcript', na=False)].empty if not files_df.empty else False

    if can_use_dashboard:
        logger.info("[green]✓ Interview is ready for dashboard viewing[/green]", extra={"markup": True})
        logger.info(f"  URL: http://localhost:3000/interviews/{interview_name}")
    else:
        logger.error("[red]✗ Interview is NOT ready for dashboard[/red]", extra={"markup": True})
        if interviews_df.empty:
            logger.info("  Reason: Interview not found in database")
        elif transcript_files_df.empty:
            logger.info("  Reason: No transcript file linked")

    if can_run_qc:
        logger.info("[green]✓ Interview is ready for QC processing[/green]", extra={"markup": True})
    else:
        logger.error("[red]✗ Interview is NOT ready for QC processing[/red]", extra={"markup": True})
        if interviews_df.empty:
            logger.info("  Reason: Interview not found in database")
        elif files_df.empty or files_df[files_df['interview_file_tags'].str.contains('transcript', na=False)].empty:
            logger.info("  Reason: No transcript in interview_files table")
            logger.info("")
            logger.info("  [bold yellow]To fix:[/bold yellow]", extra={"markup": True})
            logger.info("  You need to add the transcript to 'interview_files' table.")
            logger.info("  The transcript_quick_qc runner queries 'interview_files', not 'transcript_files'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Check if an interview exists in the database."
    )
    parser.add_argument(
        "interview_name",
        type=str,
        help="Interview name (e.g., 'PREDiCTOR-P0014HA-onsiteInterview_MD-day0002')"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to the config file (optional)",
        required=False
    )

    args = parser.parse_args()

    # Get config file
    if args.config:
        config_file = Path(args.config).resolve()
    else:
        config_file = Path("/home/dpinterview/predictor.config.ini").resolve()

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)

    logger.info(f"Using config file: {config_file}")

    check_interview(args.interview_name, config_file)
