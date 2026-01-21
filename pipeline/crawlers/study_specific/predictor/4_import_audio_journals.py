#!/usr/bin/env python
"""
Looks for audio journals under:
PHOENIX/PROTECTED/*/phone/raw/*/audio_recordings/*_sound_*.mp3
"""

import sys
from pathlib import Path

file = Path(__file__).resolve()
parent = file.parent
ROOT = None
for parent in file.parents:
    if parent.name == "dpinterview":
        ROOT = parent
sys.path.append(str(ROOT))

# remove current directory from path
try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import logging
import argparse
from typing import List, Tuple, Optional
from datetime import datetime
import multiprocessing
import json

from rich.logging import RichHandler
from rich.progress import Progress

from pipeline import core, orchestrator
from pipeline.helpers import cli, utils, db, dpdash
from pipeline.models.subjects import Subject
from pipeline.models.files import File
from pipeline.models.audio_journals import AudioJournal

MODULE_NAME = "import_audio_journals"
INSTANCE_NAME = MODULE_NAME

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.DEBUG,
    # "format": "%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s",
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def get_mindlamp_id(config_file: Path, subject_id: str, study_id: str) -> Optional[str]:
    """
    Get MindLAMP ID from subjects table optional_notes JSON.
    
    Args:
        config_file: Path to config file
        subject_id: Subject ID
        study_id: Study ID
        
    Returns:
        MindLAMP ID (e.g., "U1234567") or None if not found
    """
    query = f"""
    SELECT optional_notes 
    FROM subjects 
    WHERE subject_id = '{subject_id}' AND study_id = '{study_id}';
    """
    
    result_df = db.execute_sql(config_file=config_file, query=query)
    
    if result_df.empty:
        logger.warning(f"Subject {subject_id} not found in database")
        return None
    
    optional_notes = result_df.iloc[0]['optional_notes']
    
    if not optional_notes:
        logger.warning(f"No optional_notes for subject {subject_id}")
        return None
    
    # optional_notes is already a dict (JSONB column)
    if isinstance(optional_notes, dict):
        # Try both case variations to be safe
        mindlamp_id = optional_notes.get('MindLAMP') or optional_notes.get('MindLamp')
        
        if mindlamp_id:
            logger.info(f"Found MindLAMP ID for {subject_id}: {mindlamp_id}")
            return mindlamp_id
        else:
            logger.warning(f"MindLAMP key not found in optional_notes for {subject_id}")
            logger.debug(f"Available keys: {list(optional_notes.keys())}")
            return None
    else:
        logger.error(f"optional_notes is not a dict for {subject_id}: {type(optional_notes)}")
        return None



def fetch_journals(
    config_file: Path, subject_id: str, study_id: str
) -> List[AudioJournal]:
    """
    Fetches the AudioJournals for a given subject ID.

    Args:
        config_file (Path): The path to the config file.
        subject_id (str): The subject ID.
        study_id (str): The study ID.

    Returns:
        List[AudioJournal]: A list of AudioJournal objects.
    """
    config_params = utils.config(path=config_file, section="general")
    data_root = Path(config_params["data_root"])
    study_path: Path = data_root / "PROTECTED" / study_id
    
    # Get MindLAMP ID from database
    mindlamp_id = get_mindlamp_id(config_file, subject_id, study_id)
    
    if not mindlamp_id:
        logger.warning(f"Skipping {subject_id} - no MindLAMP ID in database")
        return []  # Return empty list, not continue
    
    # Construct path with MindLAMP ID
    audio_journal_root_path = study_path / subject_id / "phone" / "raw" / mindlamp_id / "audio_recordings"

    if not audio_journal_root_path.exists():
        logger.warning(f"No audio journal path for {subject_id} at {audio_journal_root_path}")
        return []

    # Look for audio journal files (mp3 files with _sound_ pattern)
    audio_journal_path = list(audio_journal_root_path.glob("*_sound_*.mp3"))
    
    if len(audio_journal_path) == 0:
        logger.warning(f"No audio journal found for {subject_id} at {audio_journal_root_path}")
        return []
    else:
        logger.info(f"Found {len(audio_journal_path)} audio journals for {subject_id}")

    # Get subject consent date
    subject_consent_date = Subject.get_consent_date(
        study_id=study_id, subject_id=subject_id, config_file=config_file
    )
    if subject_consent_date is None:
        logger.error(f"No consent date found for {subject_id} - skipping...")
        return []

    # Process each audio journal file
    audio_journals: List[AudioJournal] = []
    for audio_journal in audio_journal_path:
        audio_journal_basename = audio_journal.name
        # Expected format: U<id>_YYYY_MM_DD_sound_N.mp3

        audio_journal_parts = audio_journal_basename.split("_")

        # Parse date: parts[1] = year, parts[2] = month, parts[3] = day
        # Expected format: U<id>_YYYY_MM_DD_sound_N.mp3
        journal_date: datetime = datetime.strptime(
            f"{audio_journal_parts[1]}-{audio_journal_parts[2]}-{audio_journal_parts[3]}",
            "%Y-%m-%d",
        )

        journal_day = dpdash.get_days_between_dates(
            consent_date=subject_consent_date, event_date=journal_date
        )

        # Session number is the last part before .mp3
        journal_session = audio_journal_parts[-1].split(".")[0]
        journal_session = int(journal_session) + 1

        audio_journal_name = dpdash.get_dpdash_name(
            study=study_id,
            subject=subject_id,
            data_type="audioJournal",
            consent_date=subject_consent_date,
            event_date=journal_date,
        )
        audio_journal_name = f"{audio_journal_name}-session{journal_session:04d}"
        
        audio_journal_obj = AudioJournal(
            aj_path=audio_journal,
            aj_name=audio_journal_name,
            aj_datetime=journal_date,
            aj_day=journal_day,
            aj_session=journal_session,
            subject_id=subject_id,
            study_id=study_id,
        )
        audio_journals.append(audio_journal_obj)

    return audio_journals


def hash_file_worker(params: Tuple[AudioJournal, Path]) -> File:
    """
    Hashes the file and returns a File object.

    Args:
        params (Tuple[AudioJournal, Path]): A tuple containing the AudioJournal
            and the path to the config file.
    """
    journal_file, config_file = params
    with_hash = orchestrator.is_crawler_hashing_required(config_file=config_file)

    file = File(file_path=journal_file.aj_path, with_hash=with_hash)
    return file


def generate_queries(
    journals: List[AudioJournal],
    config_file: Path,
    progress: Progress,
) -> List[str]:
    """
    Generates the SQL queries to insert the interview files into the database.

    Args:
        journals (List[AudioJournal]): The list of AudioJournal objects.
        config_file (Path): The path to the configuration file.
        progress (Progress): The progress bar.
    """

    files: List[File] = []

    if orchestrator.is_crawler_hashing_required(config_file=config_file):
        logger.info("Hashing files...")
    else:
        logger.info("Skipping hashing files...")

    params = [(journal, config_file) for journal in journals]
    logger.info(f"Have {len(params)} files to hash")

    num_processes = multiprocessing.cpu_count() / 2
    logger.info(f"Using {num_processes} processes")
    with multiprocessing.Pool(processes=int(num_processes)) as pool:
        task = progress.add_task("Hashing files...", total=len(params))
        for result in pool.imap_unordered(hash_file_worker, params):
            files.append(result)
            progress.update(task, advance=1)
        progress.remove_task(task)

    sql_queries = []
    logger.info("Generating SQL queries...")
    # Insert the files
    for file in files:
        sql_queries.append(file.to_sql())

    # Insert the journals
    for journal in journals:
        sql_queries.append(journal.to_sql())

    return sql_queries


def import_journals(config_file: Path, study_id: str, progress: Progress) -> None:
    """
    Imports the Audio Journals into the database.

    Args:
        config_file (Path): The path to the configuration file.
    """

    # Get the subjects
    subjects = core.get_subject_ids(config_file=config_file, study_id=study_id)

    # Get the journals
    logger.info(f"Fetching journals for {study_id}")
    journals: List[AudioJournal] = []

    task = progress.add_task("Fetching journals for subjects", total=len(subjects))
    for subject_id in subjects:
        progress.update(
            task, advance=1, description=f"Fetching {subject_id}'s journals..."
        )
        journals.extend(
            fetch_journals(
                config_file=config_file, subject_id=subject_id, study_id=study_id
            )
        )
    progress.remove_task(task)

    logger.info(f"Found {len(journals)} journals for {study_id}")

    # Generate the SQL queries
    sql_queries = generate_queries(
        journals=journals, config_file=config_file, progress=progress
    )

    # Execute the queries
    db.execute_queries(queries=sql_queries, config_file=config_file, silent=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME, description="Import audio journals into the database."
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config file.", required=False
    )

    args = parser.parse_args()

    # Check if parseer has config file
    if args.config:
        config_file = Path(args.config).resolve()
        if not config_file.exists():
            logger.error(f"Error: Config file '{config_file}' does not exist.")
            sys.exit(1)
    else:
        if cli.confirm_action("Using default config file."):
            config_file = utils.get_config_file_path()
        else:
            sys.exit(1)
    utils.configure_logging(
        config_file=config_file, module_name=MODULE_NAME, logger=logger
    )

    console.rule(f"[bold red]{MODULE_NAME}")
    logger.info(f"Using config file: {config_file}")

    study_ids = orchestrator.get_studies(config_file=config_file)
    with utils.get_progress_bar() as progress:
        study_task = progress.add_task("Importing journals...", total=len(study_ids))
        for study_id in study_ids:
            progress.update(
                study_task,
                advance=1,
                description=f"Importing journals for {study_id}...",
            )
            import_journals(
                config_file=config_file, study_id=study_id, progress=progress
            )

    logger.info("Done!")
