#!/usr/bin/env python
"""
Simple transcript path updater when QC and transcript_files tables are already cleaned up.

This script only updates:
- interview_files table
- files table
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
from pipeline.models.files import File

MODULE_NAME = "update_transcript_path_simple"
logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def update_simple(old_path: Path, new_path: Path, config_file: Path) -> None:
    """
    Simple update when transcript_files and transcript_quick_qc are already cleaned.
    """

    if not new_path.exists():
        logger.error(f"New file does not exist: {new_path}")
        sys.exit(1)

    logger.info(f"Old path: {old_path}")
    logger.info(f"New path: {new_path}")

    # Check if old path exists in interview_files
    query = f"""
        SELECT interview_path, interview_file_tags
        FROM interview_files
        WHERE interview_file = '{old_path}';
    """
    result = db.execute_sql(config_file=config_file, query=query)

    if result.empty:
        logger.error(f"Old path not found in interview_files: {old_path}")
        sys.exit(1)

    interview_path = result.iloc[0]['interview_path']
    tags = result.iloc[0]['interview_file_tags']
    logger.info(f"Found in interview_files: {interview_path}")
    logger.info(f"Tags: {tags}")

    # Build queries
    queries = []

    # 1. Insert new file entry
    try:
        new_file = File(file_path=new_path, with_hash=True)
        queries.append(new_file.to_sql())
        logger.info(f"Will create new file entry")
    except Exception as e:
        logger.error(f"Error creating File model: {e}")
        sys.exit(1)

    # 2. Insert new interview_files entry
    queries.append(f"""
        INSERT INTO interview_files (interview_path, interview_file, interview_file_tags)
        VALUES ('{interview_path}', '{new_path}', '{tags}')
        ON CONFLICT (interview_path, interview_file) DO NOTHING;
    """)

    # 3. Delete old interview_files entry
    queries.append(f"""
        DELETE FROM interview_files
        WHERE interview_file = '{old_path}';
    """)

    # 4. Delete old file entry
    queries.append(f"""
        DELETE FROM files
        WHERE file_path = '{old_path}';
    """)

    # Execute
    logger.info("Executing updates...")
    try:
        db.execute_queries(config_file=config_file, queries=queries, show_commands=True)
        logger.info("[bold green]✓ Success![/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)

    console.rule("[bold green]Complete[/bold green]")
    logger.info("Updated tables:")
    logger.info("  ✓ files")
    logger.info("  ✓ interview_files")
    logger.info("")
    logger.info("Now re-import to transcript_files:")
    logger.info(f"  python /home/dpinterview/etc/import_single_transcript.py \\")
    logger.info(f"      {new_path} \\")
    logger.info(f"      <interview_name> \\")
    logger.info(f"      -c predictor.config.ini")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Simple path updater for interview_files and files tables."
    )
    parser.add_argument("old_path", type=str, help="Old transcript path")
    parser.add_argument("new_path", type=str, help="New transcript path")
    parser.add_argument("-c", "--config", type=str, required=False)

    args = parser.parse_args()

    if args.config:
        config_file = Path(args.config).resolve()
    else:
        config_file = Path("/home/dpinterview/predictor.config.ini").resolve()

    if not config_file.exists():
        logger.error(f"Config not found: {config_file}")
        sys.exit(1)

    logger.info(f"Using config: {config_file}")

    old_path = Path(args.old_path).resolve()
    new_path = Path(args.new_path).resolve()

    update_simple(old_path, new_path, config_file)
