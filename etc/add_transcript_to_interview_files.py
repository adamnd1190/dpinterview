#!/usr/bin/env python
"""
Add an existing transcript from transcript_files to interview_files table.

This is needed because:
- transcript_files: Used by dashboard for viewing
- interview_files: Used by QC runners for processing

Both tables need the transcript entry for full functionality.

Usage:
    python add_transcript_to_interview_files.py <interview_name>
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

from pipeline.helpers import utils, db
from pipeline.models.interview_files import InterviewFile

MODULE_NAME = "add_transcript_to_interview_files"
logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def add_to_interview_files(interview_name: str, config_file: Path) -> None:
    """
    Add transcript from transcript_files to interview_files table.

    Args:
        interview_name (str): Interview name
        config_file (Path): Config file path
    """

    logger.info(f"Looking up interview: {interview_name}")

    # Get interview_path from interview_parts table (linked to interviews via interview_name)
    query = f"""
        SELECT interview_path FROM interview_parts
        WHERE interview_name = '{interview_name}' AND is_primary = TRUE;
    """

    interview_path = db.fetch_record(config_file=config_file, query=query)

    if not interview_path:
        # Try without is_primary filter
        query = f"""
            SELECT interview_path FROM interview_parts
            WHERE interview_name = '{interview_name}'
            LIMIT 1;
        """
        interview_path = db.fetch_record(config_file=config_file, query=query)

    if not interview_path:
        logger.error(f"Interview '{interview_name}' not found in 'interview_parts' table")
        logger.error("The interview must be imported first before adding transcripts.")
        logger.error("Check that the interview exists with:")
        logger.error(f"  SELECT * FROM interviews WHERE interview_name = '{interview_name}';")
        logger.error(f"  SELECT * FROM interview_parts WHERE interview_name = '{interview_name}';")
        sys.exit(1)

    logger.info(f"Found interview_path: {interview_path}")

    # Get transcript file path from transcript_files table
    query = f"""
        SELECT transcript_file, transcript_file_tags FROM transcript_files
        WHERE identifier_name = '{interview_name}' AND identifier_type = 'interview';
    """

    result = db.execute_sql(config_file=config_file, query=query)

    if result.empty:
        logger.error(f"No transcript found in 'transcript_files' table for interview '{interview_name}'")
        logger.error("You need to import the transcript first using import_single_transcript.py")
        sys.exit(1)

    transcript_file_path = Path(result.iloc[0]['transcript_file'])
    tags = result.iloc[0]['transcript_file_tags']

    logger.info(f"Found transcript: {transcript_file_path}")
    logger.info(f"Tags: {tags}")

    # Check if file exists
    if not transcript_file_path.exists():
        logger.error(f"Transcript file not found on disk: {transcript_file_path}")
        sys.exit(1)

    # Check if already in interview_files
    query = f"""
        SELECT * FROM interview_files
        WHERE interview_path = '{interview_path}' AND interview_file = '{transcript_file_path}';
    """

    existing = db.execute_sql(config_file=config_file, query=query)

    if not existing.empty:
        logger.warning(f"Transcript already exists in interview_files table")
        logger.info("Nothing to do.")
        return

    # Add to interview_files
    logger.info("Adding transcript to interview_files table...")

    # Add "transcript" tag if not already present
    if "transcript" not in tags:
        tags = f"transcript,{tags}"

    interview_file = InterviewFile(
        interview_path=Path(interview_path),
        interview_file=transcript_file_path,
        tags=tags
    )

    sql_query = interview_file.to_sql()

    try:
        db.execute_queries(config_file=config_file, queries=[sql_query], show_commands=True)
        logger.info("[bold green]✓ Successfully added transcript to interview_files![/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"Error adding to interview_files: {e}")
        sys.exit(1)

    # Summary
    console.rule("[bold green]Success[/bold green]")
    logger.info("Transcript is now in both tables:")
    logger.info("  ✓ transcript_files (for dashboard viewing)")
    logger.info("  ✓ interview_files (for QC processing)")
    logger.info("")
    logger.info("The transcript_quick_qc runner can now process this transcript.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Add transcript to interview_files table for QC processing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python add_transcript_to_interview_files.py PREDiCTOR-P0014HA-onsiteInterview_MD-day0002

This script:
1. Looks up the interview_path from the interviews table
2. Gets the transcript path from transcript_files table
3. Adds an entry to interview_files table with the transcript tag
        """
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

    add_to_interview_files(args.interview_name, config_file)
