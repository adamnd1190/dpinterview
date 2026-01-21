#!/usr/bin/env python
"""
Update transcript file path in the database.

This is useful when you move a transcript file to a new location
and need to update the database references.
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

MODULE_NAME = "update_transcript_path"
logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def update_transcript_path(
    old_path: Path,
    new_path: Path,
    config_file: Path
) -> None:
    """
    Update transcript file path in all database tables.

    Args:
        old_path: Old transcript file path
        new_path: New transcript file path
        config_file: Config file path
    """

    # Verify new file exists
    if not new_path.exists():
        logger.error(f"New file path does not exist: {new_path}")
        sys.exit(1)

    logger.info(f"Old path: {old_path}")
    logger.info(f"New path: {new_path}")

    # Check if old path exists in database
    query = f"""
        SELECT * FROM files WHERE file_path = '{old_path}';
    """
    result = db.execute_sql(config_file=config_file, query=query)

    if result.empty:
        logger.error(f"Old path not found in database: {old_path}")
        sys.exit(1)

    logger.info(f"Found old path in database")

    # Update files table
    queries = []

    # First, update or insert the new file entry
    from pipeline.models.files import File
    try:
        new_file = File(file_path=new_path, with_hash=True)
        queries.append(new_file.to_sql())
        logger.info(f"Will insert/update new file entry")
    except Exception as e:
        logger.error(f"Error creating File model: {e}")
        sys.exit(1)

    # IMPORTANT: Update in order of foreign key dependencies
    # The constraint is: transcript_quick_qc.transcript_path -> interview_files.interview_file -> files.file_path

    # Strategy: Temporarily disable the constraint, update all tables, then re-enable
    # OR: Insert new entry first, update references, then delete old

    # 1. First, insert the new file entry into interview_files (so transcript_quick_qc can reference it)
    # We need to get the interview_path for this
    get_interview_path_query = f"""
        SELECT interview_path FROM interview_files
        WHERE interview_file = '{old_path}'
        LIMIT 1;
    """

    logger.info("Looking up interview_path...")
    interview_path_result = db.execute_sql(config_file=config_file, query=get_interview_path_query)

    if interview_path_result.empty:
        logger.error(f"Could not find interview_path for old file in interview_files table")
        sys.exit(1)

    interview_path = interview_path_result.iloc[0]['interview_path']
    interview_file_tags = interview_path_result.iloc[0]['interview_file_tags']
    logger.info(f"Found interview_path: {interview_path}")
    logger.info(f"Tags: {interview_file_tags}")

    # Insert new interview_files entry (so transcript_quick_qc can reference it)
    queries.append(f"""
        INSERT INTO interview_files (interview_path, interview_file, interview_file_tags)
        VALUES ('{interview_path}', '{new_path}', '{interview_file_tags}')
        ON CONFLICT (interview_path, interview_file) DO NOTHING;
    """)

    # 2. Now update transcript_quick_qc to reference the new path
    queries.append(f"""
        UPDATE transcript_quick_qc
        SET transcript_path = '{new_path}'
        WHERE transcript_path = '{old_path}';
    """)

    # 3. Delete the old interview_files entry (now that transcript_quick_qc points to new path)
    queries.append(f"""
        DELETE FROM interview_files
        WHERE interview_file = '{old_path}';
    """)

    # 4. Update transcript_files (references files)
    queries.append(f"""
        UPDATE transcript_files
        SET transcript_file = '{new_path}'
        WHERE transcript_file = '{old_path}';
    """)

    # 5. Finally delete old file entry
    queries.append(f"""
        DELETE FROM files
        WHERE file_path = '{old_path}';
    """)

    # Execute all queries
    logger.info("Executing database updates...")
    try:
        db.execute_queries(config_file=config_file, queries=queries, show_commands=True)
        logger.info("[bold green]✓ Successfully updated transcript path in database![/bold green]", extra={"markup": True})
    except Exception as e:
        logger.error(f"Error updating database: {e}")
        sys.exit(1)

    console.rule("[bold green]Success[/bold green]")
    logger.info("Transcript path updated in all tables:")
    logger.info("  ✓ files")
    logger.info("  ✓ transcript_files")
    logger.info("  ✓ interview_files")
    logger.info("  ✓ transcript_quick_qc")
    logger.info("")
    logger.info("You can now access the transcript on the dashboard!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Update transcript file path in database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python update_transcript_path.py \\
        /mnt/old/path/transcript.txt \\
        /mnt/new/path/transcript.txt \\
        -c predictor.config.ini
        """
    )
    parser.add_argument(
        "old_path",
        type=str,
        help="Old transcript file path"
    )
    parser.add_argument(
        "new_path",
        type=str,
        help="New transcript file path"
    )
    parser.add_argument(
        "-c", "--config",
        type=str,
        help="Path to config file",
        required=False
    )

    args = parser.parse_args()

    # Get config
    if args.config:
        config_file = Path(args.config).resolve()
    else:
        config_file = Path("/home/dpinterview/predictor.config.ini").resolve()

    if not config_file.exists():
        logger.error(f"Config file not found: {config_file}")
        sys.exit(1)

    logger.info(f"Using config file: {config_file}")

    old_path = Path(args.old_path).resolve()
    new_path = Path(args.new_path).resolve()

    update_transcript_path(old_path, new_path, config_file)
