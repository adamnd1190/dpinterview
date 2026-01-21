#!/usr/bin/env python
"""
MODIFIED: Walks through interview directories and imports the interview files into the database.

NOW SUPPORTS CRC/MD TAGGING:
- Regular interviews: onsiteInterview_MD, offsiteInterview_MD  
- Baseline: offsiteInterview_CRC_Baseline
- 6mo: offsiteInterview_CRC_6mo
- 12mo: offsiteInterview_CRC_12mo
- Cognition: onsiteInterview_CRC_cognition or offsiteInterview_CRC_cognition
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

try:
    sys.path.remove(str(parent))
except ValueError:
    pass

import argparse
import logging
import multiprocessing
import re
from datetime import date, datetime, time
from typing import Dict, List, Tuple

from rich.logging import RichHandler
from rich.progress import Progress

from pipeline import core, orchestrator
from pipeline.helpers import cli, db, dpdash, utils
from pipeline.helpers.config import config
from pipeline.models.files import File
from pipeline.models.interview_files import InterviewFile
from pipeline.models.interview_parts import InterviewParts
from pipeline.models.interviews import Interview, InterviewType

MODULE_NAME = "import_interview_files"
INSTANCE_NAME = MODULE_NAME

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.DEBUG,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def catogorize_audio_files(
    audio_files: List[Path], subject_id: str
) -> Dict[str, List[Path]]:
    """
    Categorizes the audio files into participant, interviewer, and combined audio files.
    """
    files: Dict[str, List[Path]] = {}

    files["combined"] = []
    files["participant"] = []
    files["interviewer"] = []

    uncategorized_files: List[Path] = []

    for file in audio_files:
        base_name = file.name
        base_name = base_name.split(".")[0]

        if base_name == "audio_only":
            files["combined"].append(file)
            continue

        if subject_id in file.name:
            files["participant"].append(file)
            continue

        if "interviewer" in file.name.lower():
            files["interviewer"].append(file)
            continue

        uncategorized_files.append(file)

    unknown_files: List[Path] = []
    for uncategorized_file in uncategorized_files:
        unassigned = True
        base_name = uncategorized_file.name
        base_name = base_name.split(".")[0]
        last_part = base_name.split("_")[-1]

        if (
            last_part.isdigit()
            and unassigned
            and "Audio Record" not in str(uncategorized_file)
        ):
            unassigned = False
            files["combined"].append(uncategorized_file)

        if len(base_name.split("_")) == 1 and unassigned:
            pattern = r"audio\d+"

            if re.match(pattern, base_name):
                unassigned = False
                files["combined"].append(uncategorized_file)

        if unassigned:
            unknown_files.append(uncategorized_file)

    files["uncategorized"] = []
    if len(unknown_files) == 1:
        if len(files["participant"]) == 0:
            files["participant"] = unknown_files
        elif len(files["interviewer"]) == 0:
            files["interviewer"] = unknown_files
        elif len(files["combined"]) == 0:
            files["combined"] = unknown_files
        else:
            files["uncategorized"] = unknown_files
    else:
        files["uncategorized"] = unknown_files

    return files


def fetch_interview_files(interview_part: InterviewParts) -> List[InterviewFile]:
    """
    Fetches the interview files for a given interview.
    """
    interview_files: List[InterviewFile] = []

    dp_dash_dict = dpdash.parse_dpdash_name(interview_part.interview_name)
    subject_id = dp_dash_dict["subject"]
    if not isinstance(subject_id, str):
        logger.error(f"Could not parse subject ID from {interview_part.interview_name}")
        raise ValueError(f"Could not parse subject ID from {interview_part.interview_name}")
    
    interview_path = interview_part.interview_path

    if interview_path.is_file():
        interview_files.append(
            InterviewFile(
                interview_path=interview_path,
                interview_file=interview_path,
                tags="audio,combined",
            )
        )
        return interview_files

    files = [f for f in interview_path.iterdir() if f.is_file()]
    audio_record_dir = interview_path / "Audio Record"
    if audio_record_dir.exists():
        files.extend([f for f in audio_record_dir.iterdir() if f.is_file()])

    audio_files: List[Path] = []
    video_files: List[Path] = []
    for file in files:
        base_name = file.name
        base_name_parts = base_name.split(".")
        if base_name[0] == ".":
            continue
        if base_name_parts[-1].lower() in ["mp4", "mkv", "mov"]:
            video_files.append(file)
        elif base_name_parts[-1] == "m4a":
            audio_files.append(file)

    categorized_audio_files = catogorize_audio_files(
        audio_files=audio_files, subject_id=subject_id
    )

    for tag, audio_files in categorized_audio_files.items():
        for audio_file in audio_files:
            tags = f"audio,{tag}"
            if "Audio Record" in audio_file.parts:
                tags += ",diarized"
            interview_file = InterviewFile(
                interview_path=interview_path,
                interview_file=audio_file,
                tags=tags,
            )
            interview_files.append(interview_file)

    for video_file in video_files:
        interview_file = InterviewFile(
            interview_path=interview_path, interview_file=video_file, tags="video"
        )
        interview_files.append(interview_file)

    return interview_files


def handle_multi_part_interviews(
    interview_parts: List[InterviewParts]
) -> List[InterviewParts]:
    """
    Sorts and groups the interviews by day, fixing part numbers chronologically.
    """
    interview_parts.sort(key=lambda x: x.interview_datetime)

    for idx, interview in enumerate(interview_parts):
        if idx == 0:
            interview.interview_part = 1
        else:
            prev_interview = interview_parts[idx - 1]
            current_interview_date = interview.interview_datetime.date()
            prev_interview_date = prev_interview.interview_datetime.date()

            if prev_interview_date == current_interview_date:
                interview.interview_part = prev_interview.interview_part + 1
            else:
                interview.interview_part = 1

    return interview_parts


def fetch_interviews(
    config_file: Path, subject_id: str, study_id: str
) -> List[InterviewParts]:
    """
    Fetches the interviews for a given subject ID.
    
    MODIFIED TO SUPPORT CRC/MD TAGGING
    """
    config_params = config(path=config_file, section="general")
    data_root = Path(config_params["data_root"])

    study_path: Path = data_root / "PROTECTED" / study_id
    interview_types: List[InterviewType] = [InterviewType.ONSITE, InterviewType.OFFSITE]

    interview_parts: List[InterviewParts] = []
    for interview_type in interview_types:
        interview_type_path = study_path / subject_id / interview_type.value / "raw"

        if not interview_type_path.exists():
            logger.warning(
                f"{subject_id}: Could not find {interview_type.value} interviews: {interview_type_path} does not exist."
            )
            continue
            
        interview_dirs = [d for d in interview_type_path.iterdir() if d.is_dir()]
        interview_type_parts: List[InterviewParts] = []
        
        for interview_dir in interview_dirs:
            base_name = interview_dir.name
            parts = base_name.split("_")

            # MODIFIED: Detect CRC vs MD interview types
            interview_tag = "MD"  # Default to MD (regular interviews)
            date_part_idx = 1  # Default for regular format
            time_start_idx = 2  # Default time starts at index 2
            
            if len(parts) >= 3:
                second_part = parts[1].lower()
                
                # Map folder names to CRC subtypes
                crc_subtypes = {
                    "baseline": "CRC_Baseline",
                    "6mo": "CRC_6mo",
                    "12mo": "CRC_12mo",
                    "cognition": "CRC_cognition"
                }
                
                if second_part in crc_subtypes:
                    interview_tag = crc_subtypes[second_part]
                    date_part_idx = 2
                    time_start_idx = 3

            try:
                date_dt = datetime.strptime(parts[date_part_idx], "%d%b%Y").date()
            except ValueError:
                try:
                    date_dt = datetime.strptime(parts[date_part_idx], "%y%m%d").date()
                except ValueError:
                    try:
                        date_dt = datetime.strptime(parts[date_part_idx], "%Y%m%d").date()
                    except ValueError:
                        logger.error(f"Could not parse date for {base_name}")
                        continue

            if len(parts) >= time_start_idx + 3:
                try:
                    time_dt = datetime.strptime(
                        f"{parts[time_start_idx]}:{parts[time_start_idx+1]}:{parts[time_start_idx+2]}", 
                        "%H:%M:%S"
                    ).time()
                except ValueError:
                    logger.error(f"Could not parse time for {base_name}")
                    time_dt = datetime.strptime("00:00:00", "%H:%M:%S").time()
            else:
                logger.info(f"No time info in {base_name}, using 00:00:00")
                time_dt = datetime.strptime("00:00:00", "%H:%M:%S").time()

            interview_datetime = datetime.combine(date_dt, time_dt)
            actual_interview_datetime = interview_datetime

            consent_date_s = core.get_consent_date_from_subject_id(
                config_file=config_file, subject_id=subject_id, study_id=study_id
            )
            if consent_date_s is None:
                logger.warning(f"Could not find consent date for {subject_id}")
                continue
            consent_date = datetime.strptime(consent_date_s, "%Y-%m-%d")

            # MODIFIED: Build data_type with MD/CRC tag
            base_data_type = "onsiteInterview" if interview_type == InterviewType.ONSITE else "offsiteInterview"
            data_type = f"{base_data_type}_{interview_tag}"

            interview_name = dpdash.get_dpdash_name(
                study=study_id,
                subject=subject_id,
                data_type=data_type,
                consent_date=consent_date,
                event_date=interview_datetime,
            )
            
            interview_day = dpdash.get_days_between_dates(
                consent_date=consent_date, event_date=interview_datetime
            )

            interview_part = InterviewParts(
                interview_name=interview_name,
                interview_path=interview_dir,
                interview_day=interview_day,
                interview_part=1,
                interview_datetime=actual_interview_datetime,
            )

            interview_type_parts.append(interview_part)

        wav_files = [
            f for f in interview_type_path.glob("*.WAV")
            if not f.name.startswith(".check_sum_")
        ]

        for wav_file in wav_files:
            interview_datetime_str = wav_file.stem
            try:
                actual_interview_datetime = datetime.strptime(interview_datetime_str, "%Y%m%d%H%M%S")
                interview_datetime = actual_interview_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                logger.warning(f"Could not parse date and time from {wav_file}. Skipping...")
                continue

            consent_date_s = core.get_consent_date_from_subject_id(
                config_file=config_file, subject_id=subject_id, study_id=study_id
            )

            if consent_date_s is None:
                logger.warning(f"Could not find consent date for {subject_id}")
                continue
            consent_date = datetime.strptime(consent_date_s, "%Y-%m-%d")

            # WAV files are always MD type
            base_data_type = "onsiteInterview" if interview_type == InterviewType.ONSITE else "offsiteInterview"
            data_type = f"{base_data_type}_MD"
            
            interview_name = dpdash.get_dpdash_name(
                study=study_id,
                subject=subject_id,
                data_type=data_type,
                consent_date=consent_date,
                event_date=interview_datetime,
            )
            interview_day = dpdash.get_days_between_dates(
                consent_date=consent_date, event_date=interview_datetime
            )

            interview_part = InterviewParts(
                interview_name=interview_name,
                interview_path=wav_file,
                interview_day=interview_day,
                interview_part=1,
                interview_datetime=actual_interview_datetime,
            )

            interview_type_parts.append(interview_part)

        interview_type_parts = handle_multi_part_interviews(interview_type_parts)
        interview_parts.extend(interview_type_parts)

    return interview_parts


def hash_file_worker(params: Tuple[InterviewFile, Path]) -> File:
    """Hashes the file and returns a File object."""
    interview_file, config_file = params
    with_hash = orchestrator.is_crawler_hashing_required(config_file=config_file)
    file = File(file_path=interview_file.interview_file, with_hash=with_hash)
    return file


def generate_queries(
    interview_parts: List[InterviewParts],
    interview_files: List[InterviewFile],
    config_file: Path,
    progress: Progress,
) -> List[str]:
    """Generates the SQL queries to insert the interview files into the database."""
    files: List[File] = []

    if orchestrator.is_crawler_hashing_required(config_file=config_file):
        logger.info("Hashing files...")
    else:
        logger.info("Skipping hashing files...")

    params = [(interview_file, config_file) for interview_file in interview_files]

    num_processes = multiprocessing.cpu_count() / 2
    logger.info(f"Using {num_processes} processes")
    with multiprocessing.Pool(processes=int(num_processes)) as pool:
        task = progress.add_task("Hashing files...", total=len(interview_files))
        for result in pool.imap_unordered(hash_file_worker, params):
            files.append(result)
            progress.update(task, advance=1)
        progress.remove_task(task)

    sql_queries = []
    logger.info("Generating SQL queries...")
    
    for file in files:
        sql_queries.append(file.to_sql())

    for interview_part in interview_parts:
        dp_dash_dict = dpdash.parse_dpdash_name(interview_part.interview_name)
        subject_id = dp_dash_dict["subject"]
        study_id = dp_dash_dict["study"]
        data_type = dp_dash_dict["data_type"]

        data_type_parts = utils.camel_case_split(data_type)
        if data_type_parts[0].lower() == 'onsite':
            interview_type = InterviewType.ONSITE
        elif data_type_parts[0].lower() == 'offsite':
            interview_type = InterviewType.OFFSITE
        else:
            interview_type = InterviewType.ONSITE
            
        interview = Interview(
            interview_name=interview_part.interview_name,
            interview_type=interview_type,
            subject_id=subject_id,
            study_id=study_id,
        )

        sql_queries.append(interview.to_sql())

    for interview_part in interview_parts:
        sql_queries.append(interview_part.to_sql())

    for interview_file in interview_files:
        sql_queries.append(interview_file.to_sql())

    return sql_queries


def import_interviews(config_file: Path, study_id: str, progress: Progress) -> None:
    """Imports the interviews into the database."""
    subjects = core.get_subject_ids(config_file=config_file, study_id=study_id)

    logger.info(f"Fetching interviews for {study_id}")
    interview_parts: List[InterviewParts] = []

    task = progress.add_task("Fetching interviews for subjects", total=len(subjects))
    for subject_id in subjects:
        progress.update(task, advance=1, description=f"Fetching {subject_id}'s interviews...")
        interview_parts.extend(
            fetch_interviews(config_file=config_file, subject_id=subject_id, study_id=study_id)
        )
    progress.remove_task(task)

    logger.info("Fetching interview files...")
    interview_files: List[InterviewFile] = []

    task = progress.add_task("Fetching interview files...", total=len(interview_parts))
    for interview_part in interview_parts:
        progress.update(task, advance=1)
        interview_files.extend(fetch_interview_files(interview_part=interview_part))
    progress.remove_task(task)

    sql_queries = generate_queries(
        interview_parts=interview_parts,
        interview_files=interview_files,
        config_file=config_file,
        progress=progress,
    )

    db.execute_queries(config_file=config_file, queries=sql_queries, show_commands=False)


def mark_unique_interviews_as_primary(config_file: Path, study_id: str) -> None:
    """Mark unique interviews as primary."""
    query = f"""
    WITH duplicate_groups AS (
        SELECT ip.interview_name, ip.interview_day
        FROM public.interview_parts ip
        JOIN public.interviews i ON ip.interview_name = i.interview_name
        WHERE i.study_id = '{study_id}'
        GROUP BY ip.interview_name, ip.interview_day
        HAVING COUNT(*) > 1
    )
    UPDATE public.interview_parts ip
    SET is_primary = true
    FROM public.interviews i
    WHERE ip.interview_name = i.interview_name
    AND i.study_id = '{study_id}'
    AND (ip.interview_name, ip.interview_day) NOT IN (
        SELECT dg.interview_name, dg.interview_day
        FROM duplicate_groups dg
    );
    """

    db.execute_queries(config_file=config_file, queries=[query])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME, description="Gather metadata for files."
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config file.", required=False
    )

    args = parser.parse_args()

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
        study_task = progress.add_task("Importing interviews...", total=len(study_ids))
        for study_id in study_ids:
            progress.update(
                study_task,
                advance=1,
                description=f"Importing interviews for {study_id}...",
            )
            import_interviews(
                config_file=config_file, study_id=study_id, progress=progress
            )
            logger.info(f"Marking unique interviews as primary for {study_id}")
            mark_unique_interviews_as_primary(
                config_file=config_file, study_id=study_id
            )

    logger.info("[bold green]Done!", extra={"markup": True})