#!/usr/bin/env python
"""
Simple script to import a single transcript file into the database for testing.

Usage:
    python import_single_transcript.py <transcript_file_path> <interview_name>

Example:
    python import_single_transcript.py \
        /mnt/PREDiCTOR/AV_TEMP/redac_temp/transcription_outputs/PREDiCTOR-P0014HA-onsiteInterview_MD-day0002_speaker_turns_diarlm.txt \
        PREDiCTOR-P0014HA-onsiteInterview_MD-day0002
"""

import sys
from pathlib import Path

# Setup path to import pipeline modules
file = Path(__file__).resolve()
ROOT = None
for parent in file.parents:
    if parent.name == "dpinterview":
        ROOT = parent
        break
sys.path.append(str(ROOT))

import logging
import argparse
from rich.logging import RichHandler

from pipeline.helpers import utils, db
from pipeline.models.files import File
from pipeline.models.transcript_files import TranscriptFile

MODULE_NAME = "import_single_transcript"
logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def import_transcript(
    transcript_path: Path,
    interview_name: str,
    config_file: Path,
    tags: str = "diarlm"
) -> None:
    """
    Import a single transcript file into the database.

    Args:
        transcript_path (Path): Path to the transcript file
        interview_name (str): Interview identifier name (e.g., "PREDiCTOR-P0014HA-onsiteInterview_MD-day0002")
        config_file (Path): Path to config file
        tags (str): Comma-separated tags for the transcript (default: "diarlm")

    Returns:
        None
    """
    # Validate file exists
    if not transcript_path.exists():
        logger.error(f"Transcript file not found: {transcript_path}")
        sys.exit(1)

    logger.info(f"Importing transcript: {transcript_path}")
    logger.info(f"Interview name: {interview_name}")
    logger.info(f"Tags: {tags}")

    # Create File model (tracks file metadata)
    try:
        file = File(file_path=transcript_path, with_hash=True)
        logger.info(f"Created File model: {file.file_name} ({file.file_size_mb:.2f} MB)")
    except Exception as e:
        logger.error(f"Error creating File model: {e}")
        sys.exit(1)

    # Create TranscriptFile model (links transcript to interview)
    transcript_file = TranscriptFile(
        transcript_file=transcript_path,
        identifier_name=interview_name,
        identifier_type="interview",
        tags=tags
    )
    logger.info(f"Created TranscriptFile model: {transcript_file}")

    # Generate SQL queries
    sql_queries = [
        file.to_sql(),
        transcript_file.to_sql()
    ]

    # Execute queries
    logger.info("Inserting into database...")
    try:
        db.execute_queries(queries=sql_queries, config_file=config_file, show_commands=True)
        logger.info("[bold green]✓ Successfully imported transcript![/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"Error inserting into database: {e}")
        sys.exit(1)

    # Show next steps
    console.rule("[bold blue]Next Steps[/bold blue]")
    logger.info("1. The transcript is now in the database")
    logger.info("2. Access it on the dashboard at:")
    logger.info(f"   [cyan]http://localhost:3000/interviews/{interview_name}[/cyan]", extra={"markup": True})
    logger.info("3. Or via the old server:")
    logger.info(f"   [cyan]http://localhost:45000/transcripts/view/{interview_name}[/cyan]", extra={"markup": True})
    logger.info("")
    logger.info("Note: The transcript_quick_qc runner will automatically process this transcript")
    logger.info("if you have runner 21_transcript_quick_qc.py running.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Import a single transcript file into the database for testing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python import_single_transcript.py \\
      /path/to/transcript.txt \\
      PREDiCTOR-P0014HA-onsiteInterview_MD-day0002

  python import_single_transcript.py \\
      /path/to/transcript.txt \\
      PREDiCTOR-P0014HA-onsiteInterview_MD-day0002 \\
      --tags "diarlm,test"
        """
    )
    parser.add_argument(
        "transcript_path",
        type=str,
        help="Path to the transcript file"
    )
    parser.add_argument(
        "interview_name",
        type=str,
        help="Interview identifier name (e.g., 'PREDiCTOR-P0014HA-onsiteInterview_MD-day0002')"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to the config file (optional, defaults to predictor.config.ini)",
        required=False
    )
    parser.add_argument(
        "-t", "--tags",
        type=str,
        default="diarlm",
        help="Comma-separated tags for the transcript (default: 'diarlm')"
    )

    args = parser.parse_args()

    # Get config file
    if args.config:
        config_file = Path(args.config).resolve()
        if not config_file.exists():
            logger.error(f"Config file not found: {config_file}")
            sys.exit(1)
    else:
        # Default to predictor config
        config_file = Path("/home/dpinterview/predictor.config.ini").resolve()
        if not config_file.exists():
            logger.error(f"Default config file not found: {config_file}")
            logger.error("Please specify a config file with -c/--config")
            sys.exit(1)

    logger.info(f"Using config file: {config_file}")

    # Convert paths
    transcript_path = Path(args.transcript_path).resolve()

    # Import transcript
    import_transcript(
        transcript_path=transcript_path,
        interview_name=args.interview_name,
        config_file=config_file,
        tags=args.tags
    )
