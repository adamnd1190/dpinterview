#!/usr/bin/env python
"""
Tool to delete interview records from database and processed folders.

This tool helps clean up interview data for reprocessing by:
1. Deleting database records (all tables or specific tables)
2. Deleting processed files/folders

Safety features:
- Shows what will be deleted before executing
- Requires typing "delete" to confirm
- Supports both raw and dpdash interview names
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
from rich.prompt import Prompt, Confirm
import shutil

from pipeline.helpers import utils, db
from pipeline import orchestrator

MODULE_NAME = "cleanup_interview"
logger = logging.getLogger(MODULE_NAME)

# Temp root for intermediate processed data
TEMP_ROOT = Path("/home/dpinterview/temp_root")
logargs = {
    "level": logging.INFO,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = Console()


# Tables that reference interviews (in CORRECT order for safe deletion considering FK constraints)
# Order matters: delete children before parents to avoid FK violations
DATABASE_TABLES = [
    "pdf_reports",              # References interviews
    "load_openface",            # References openface_qc
    "openface_qc",              # References openface
    "openface",                 # References video_streams
    "video_streams",            # References video_quick_qc (via video_path)
    "video_qqc",                # video_quick_qc references decrypted_files
    "decrypted_files",          # References interview_files
    "transcript_quick_qc",      # References interview_files
    "llm_speaker_identification",  # References transcript_files
    "transcript_files",         # References files
    "interview_files",          # References interview_parts and files
    "exported_assets",          # References interview_parts
    "interview_parts",          # References interviews
    "ffprobe_metadata_audio",   # Standalone
    "ffprobe_metadata_video",   # Standalone
    "ffprobe_metadata",         # Standalone
    "interviews",               # Parent table
]


def normalize_interview_name(interview_name: str) -> dict:
    """
    Parse interview name and extract components.

    Handles both formats:
    - Raw: PREDiCTOR_P0002HA_interviewVideoFile_onsite_day0032
    - DPDash: PREDiCTOR-P0002HA-onsiteInterview_MD-day0032

    Returns dict with: study_id, subject_id, interview_type, day
    """

    # Try DPDash format first: STUDY-SUBJECT-TYPE-dayXXXX
    if "-" in interview_name and "day" in interview_name:
        parts = interview_name.split("-")
        if len(parts) >= 4:
            study_id = parts[0]
            subject_id = parts[1]
            interview_type_raw = parts[2]
            day_part = parts[3]

            # Extract day number
            day_num = int(day_part.replace("day", ""))

            return {
                "study_id": study_id,
                "subject_id": subject_id,
                "interview_type": interview_type_raw,
                "day": day_num,
                "format": "dpdash",
                "pattern": f"{study_id}-{subject_id}-%day{day_num:04d}%"
            }

    # Try raw format: STUDY_SUBJECT_*_TYPE_dayXXXX
    if "_" in interview_name and "day" in interview_name:
        parts = interview_name.split("_")
        if len(parts) >= 5:
            study_id = parts[0]
            subject_id = parts[1]
            interview_type = parts[3]
            day_part = parts[4]

            day_num = int(day_part.replace("day", ""))

            return {
                "study_id": study_id,
                "subject_id": subject_id,
                "interview_type": interview_type,
                "day": day_num,
                "format": "raw",
                "pattern": f"{study_id}-{subject_id}-%day{day_num:04d}%"
            }

    # If can't parse, use as-is for exact match
    return {
        "pattern": interview_name,
        "format": "exact"
    }


def get_flexible_path_pattern(interview_name: str) -> str:
    """
    Generate a flexible LIKE pattern for matching paths that contain interview name variants.

    For example, PREDiCTOR-P0076MA-offsiteInterview_MD-day0215 becomes:
    %PREDiCTOR-P0076MA-offsiteInterview%day0215%

    This matches variants like:
    - PREDiCTOR-P0076MA-offsiteInterview_video_subject-day0215
    - PREDiCTOR-P0076MA-offsiteInterview_video_interviewer-day0215
    - PREDiCTOR-P0076MA-offsiteInterview_MD-day0215
    """
    info = normalize_interview_name(interview_name)

    if info["format"] == "dpdash":
        # Extract the interview type prefix (e.g., "offsiteInterview" from "offsiteInterview_MD")
        interview_type_prefix = info["interview_type"].split("_")[0]
        day_num = info["day"]
        # Pattern: %STUDY-SUBJECT-TYPE_PREFIX%dayXXXX%
        return f"%{info['study_id']}-{info['subject_id']}-{interview_type_prefix}%day{day_num:04d}%"
    elif info["format"] == "raw":
        day_num = info["day"]
        return f"%{info['study_id']}-{info['subject_id']}%day{day_num:04d}%"
    else:
        return f"%{interview_name}%"


def find_related_interviews(config_file: Path, interview_name: str) -> list:
    """
    Find all interviews matching the pattern (including variants with _part suffixes).

    Handles both:
    - DPDash names: PREDiCTOR-P0002HA-onsiteInterview_MD-day0032
    - Raw names: PREDiCTOR_P0002HA_interviewVideoFile_onsite_day0032
      (raw names are stored as interview_path in interview_parts, need to resolve)
    """

    info = normalize_interview_name(interview_name)
    pattern = info["pattern"]

    # First, try to find by dpdash pattern in interviews table
    query = f"""
        SELECT DISTINCT interview_name
        FROM interviews
        WHERE interview_name LIKE '{pattern}'
        ORDER BY interview_name;
    """

    result = db.execute_sql(config_file=config_file, query=query)

    if not result.empty:
        return result["interview_name"].tolist()

    # If no match, the input might be a raw name
    # Raw names appear at the end of interview_path in interview_parts
    # e.g., interview_path ends with: /PREDiCTOR_P0002HA_interviewVideoFile_onsite_day0032.mp4
    logger.info("No exact match in interviews table. Checking interview_parts for raw name...")

    # Try to find by raw name in interview_parts.interview_path
    raw_query = f"""
        SELECT DISTINCT interview_name
        FROM interview_parts
        WHERE interview_path LIKE '%{interview_name}%'
        ORDER BY interview_name;
    """

    raw_result = db.execute_sql(config_file=config_file, query=raw_query)

    if not raw_result.empty:
        logger.info(f"Found via raw name lookup in interview_parts")
        return raw_result["interview_name"].tolist()

    return []


def show_database_preview(config_file: Path, interview_names: list, table: str = None):
    """
    Show what will be deleted from database.
    """

    interview_list = "', '".join(interview_names)

    tables_to_check = [table] if table else DATABASE_TABLES

    table_summary = Table(title="Database Records to Delete")
    table_summary.add_column("Table", style="cyan")
    table_summary.add_column("Records", justify="right", style="yellow")
    table_summary.add_column("Sample", style="dim")

    total_records = 0

    for tbl in tables_to_check:
        # Determine the column name to search
        if tbl == "interviews":
            col = "interview_name"
        elif tbl == "interview_parts":
            col = "interview_name"
        elif tbl == "interview_files":
            # Join to get interview_name
            query = f"""
                SELECT COUNT(*) as count
                FROM interview_files
                INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                WHERE interview_parts.interview_name IN ('{interview_list}');
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                # Get sample
                sample_query = f"""
                    SELECT interview_file
                    FROM interview_files
                    INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                    WHERE interview_parts.interview_name IN ('{interview_list}')
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["interview_file"]).name if not sample_result.empty else ""

                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl == "transcript_files":
            query = f"""
                SELECT COUNT(*) as count
                FROM transcript_files
                WHERE identifier_name IN ('{interview_list}') AND identifier_type = 'interview';
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                sample_query = f"""
                    SELECT transcript_file
                    FROM transcript_files
                    WHERE identifier_name IN ('{interview_list}') AND identifier_type = 'interview'
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["transcript_file"]).name if not sample_result.empty else ""

                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl == "transcript_quick_qc":
            # References interview_files
            query = f"""
                SELECT COUNT(*) as count
                FROM transcript_quick_qc
                INNER JOIN interview_files ON transcript_quick_qc.transcript_path = interview_files.interview_file
                INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                WHERE interview_parts.interview_name IN ('{interview_list}');
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
            continue
        elif tbl == "video_streams":
            # Chain: video_streams.video_path -> video_quick_qc.video_path -> decrypted_files.destination_path
            # decrypted_files.source_path -> interview_files.interview_file
            query = f"""
                SELECT COUNT(*) as count
                FROM video_streams
                WHERE video_path IN (
                    SELECT destination_path FROM decrypted_files
                    WHERE source_path IN (
                        SELECT interview_file FROM interview_files
                        INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                        WHERE interview_parts.interview_name IN ('{interview_list}')
                    )
                );
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
            continue
        elif tbl == "video_qqc":
            # video_quick_qc.video_path -> decrypted_files.destination_path
            # decrypted_files.source_path -> interview_files.interview_file
            query = f"""
                SELECT COUNT(*) as count
                FROM video_quick_qc
                WHERE video_path IN (
                    SELECT destination_path FROM decrypted_files
                    WHERE source_path IN (
                        SELECT interview_file FROM interview_files
                        INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                        WHERE interview_parts.interview_name IN ('{interview_list}')
                    )
                );
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
            continue
        elif tbl == "decrypted_files":
            # decrypted_files.source_path references interview_files.interview_file
            query = f"""
                SELECT COUNT(*) as count
                FROM decrypted_files
                WHERE source_path IN (
                    SELECT interview_file FROM interview_files
                    INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                    WHERE interview_parts.interview_name IN ('{interview_list}')
                );
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
            continue
        elif tbl == "exported_assets":
            # Has interview_name column directly
            query = f"""
                SELECT COUNT(*) as count
                FROM exported_assets
                WHERE interview_name IN ('{interview_list}');
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
            continue
        elif tbl == "ffprobe_metadata_audio":
            # Path-based PK: fma_source_path contains interview name variants
            # Use flexible pattern to match video_subject, video_interviewer, MD, etc.
            conditions = " OR ".join([f"fma_source_path LIKE '{get_flexible_path_pattern(name)}'" for name in interview_names])
            query = f"""
                SELECT COUNT(*) as count
                FROM ffprobe_metadata_audio
                WHERE {conditions};
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                sample_query = f"""
                    SELECT fma_source_path
                    FROM ffprobe_metadata_audio
                    WHERE {conditions}
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["fma_source_path"]).name if not sample_result.empty else ""
                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl == "ffprobe_metadata_video":
            # Path-based PK: fmv_source_path contains interview name variants
            conditions = " OR ".join([f"fmv_source_path LIKE '{get_flexible_path_pattern(name)}'" for name in interview_names])
            query = f"""
                SELECT COUNT(*) as count
                FROM ffprobe_metadata_video
                WHERE {conditions};
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                sample_query = f"""
                    SELECT fmv_source_path
                    FROM ffprobe_metadata_video
                    WHERE {conditions}
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["fmv_source_path"]).name if not sample_result.empty else ""
                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl == "ffprobe_metadata":
            # Path-based PK: fm_source_path contains interview name variants
            conditions = " OR ".join([f"fm_source_path LIKE '{get_flexible_path_pattern(name)}'" for name in interview_names])
            query = f"""
                SELECT COUNT(*) as count
                FROM ffprobe_metadata
                WHERE {conditions};
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                sample_query = f"""
                    SELECT fm_source_path
                    FROM ffprobe_metadata
                    WHERE {conditions}
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["fm_source_path"]).name if not sample_result.empty else ""
                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl == "openface":
            # openface.vs_path references video_streams.vs_path
            # Follow the FK chain: video_streams -> decrypted_files -> interview_files -> interview_parts
            query = f"""
                SELECT COUNT(*) as count
                FROM openface
                WHERE vs_path IN (
                    SELECT vs_path FROM video_streams
                    WHERE video_path IN (
                        SELECT destination_path FROM decrypted_files
                        WHERE source_path IN (
                            SELECT interview_file FROM interview_files
                            INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                            WHERE interview_parts.interview_name IN ('{interview_list}')
                        )
                    )
                );
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                sample_query = f"""
                    SELECT vs_path
                    FROM openface
                    WHERE vs_path IN (
                        SELECT vs_path FROM video_streams
                        WHERE video_path IN (
                            SELECT destination_path FROM decrypted_files
                            WHERE source_path IN (
                                SELECT interview_file FROM interview_files
                                INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                                WHERE interview_parts.interview_name IN ('{interview_list}')
                            )
                        )
                    )
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["vs_path"]).name if not sample_result.empty else ""
                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl == "openface_qc":
            # openface_qc.of_processed_path references openface.of_processed_path
            # Follow the FK chain through openface
            query = f"""
                SELECT COUNT(*) as count
                FROM openface_qc
                WHERE of_processed_path IN (
                    SELECT of_processed_path FROM openface
                    WHERE vs_path IN (
                        SELECT vs_path FROM video_streams
                        WHERE video_path IN (
                            SELECT destination_path FROM decrypted_files
                            WHERE source_path IN (
                                SELECT interview_file FROM interview_files
                                INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                                WHERE interview_parts.interview_name IN ('{interview_list}')
                            )
                        )
                    )
                );
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                sample_query = f"""
                    SELECT of_processed_path
                    FROM openface_qc
                    WHERE of_processed_path IN (
                        SELECT of_processed_path FROM openface
                        WHERE vs_path IN (
                            SELECT vs_path FROM video_streams
                            WHERE video_path IN (
                                SELECT destination_path FROM decrypted_files
                                WHERE source_path IN (
                                    SELECT interview_file FROM interview_files
                                    INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                                    WHERE interview_parts.interview_name IN ('{interview_list}')
                                )
                            )
                        )
                    )
                    LIMIT 1;
                """
                sample_result = db.execute_sql(config_file=config_file, query=sample_query)
                sample = Path(sample_result.iloc[0]["of_processed_path"]).name if not sample_result.empty else ""
                table_summary.add_row(tbl, str(count), sample)
                total_records += count
            continue
        elif tbl in ["manual_qc", "pdf_reports", "llm_speaker_identification"]:
            # These have interview_name column
            try:
                query = f"""
                    SELECT COUNT(*) as count
                    FROM {tbl}
                    WHERE interview_name IN ('{interview_list}');
                """
                result = db.execute_sql(config_file=config_file, query=query)
                count = result.iloc[0]["count"] if not result.empty else 0
            except:
                count = 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
            continue
        else:
            col = "interview_name"

        # Generic query for tables with interview_name column
        try:
            query = f"""
                SELECT COUNT(*) as count
                FROM {tbl}
                WHERE {col} IN ('{interview_list}');
            """
            result = db.execute_sql(config_file=config_file, query=query)
            count = result.iloc[0]["count"] if not result.empty else 0

            if count > 0:
                table_summary.add_row(tbl, str(count), "")
                total_records += count
        except Exception as e:
            logger.debug(f"Skipping table {tbl}: {e}")
            continue

    console.print(table_summary)
    console.print(f"\n[bold yellow]Total records to delete: {total_records}[/bold yellow]")

    return total_records


def delete_database_records(config_file: Path, interview_names: list, table: str = None, dry_run: bool = False):
    """
    Delete records from database tables.

    Handles different table types:
    - Tables with interview_name column
    - Tables that join through interview_parts
    - Tables with path-based primary keys (need LIKE queries)
    """

    interview_list = "', '".join(interview_names)
    tables_to_delete = [table] if table else DATABASE_TABLES

    queries = []

    for tbl in tables_to_delete:
        if tbl == "interview_files":
            # Delete via join through interview_parts
            queries.append(f"""
                DELETE FROM interview_files
                WHERE interview_path IN (
                    SELECT interview_path FROM interview_parts
                    WHERE interview_name IN ('{interview_list}')
                );
            """)
        elif tbl == "transcript_files":
            queries.append(f"""
                DELETE FROM transcript_files
                WHERE identifier_name IN ('{interview_list}') AND identifier_type = 'interview';
            """)
        elif tbl == "transcript_quick_qc":
            queries.append(f"""
                DELETE FROM transcript_quick_qc
                WHERE transcript_path IN (
                    SELECT interview_file FROM interview_files
                    INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                    WHERE interview_parts.interview_name IN ('{interview_list}')
                );
            """)
        elif tbl == "video_streams":
            # Chain: video_streams.video_path -> video_quick_qc.video_path -> decrypted_files.destination_path
            # decrypted_files.source_path -> interview_files.interview_file
            queries.append(f"""
                DELETE FROM video_streams
                WHERE video_path IN (
                    SELECT destination_path FROM decrypted_files
                    WHERE source_path IN (
                        SELECT interview_file FROM interview_files
                        INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                        WHERE interview_parts.interview_name IN ('{interview_list}')
                    )
                );
            """)
        elif tbl == "video_qqc":
            # video_quick_qc.video_path -> decrypted_files.destination_path
            # decrypted_files.source_path -> interview_files.interview_file
            queries.append(f"""
                DELETE FROM video_quick_qc
                WHERE video_path IN (
                    SELECT destination_path FROM decrypted_files
                    WHERE source_path IN (
                        SELECT interview_file FROM interview_files
                        INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                        WHERE interview_parts.interview_name IN ('{interview_list}')
                    )
                );
            """)
        elif tbl == "decrypted_files":
            # decrypted_files.source_path references interview_files.interview_file
            queries.append(f"""
                DELETE FROM decrypted_files
                WHERE source_path IN (
                    SELECT interview_file FROM interview_files
                    INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                    WHERE interview_parts.interview_name IN ('{interview_list}')
                );
            """)
        elif tbl == "exported_assets":
            # Has interview_name column directly
            queries.append(f"""
                DELETE FROM exported_assets
                WHERE interview_name IN ('{interview_list}');
            """)
        elif tbl == "ffprobe_metadata_audio":
            # Path-based PK: fma_source_path contains interview name variants
            # Use flexible pattern to match video_subject, video_interviewer, MD, etc.
            conditions = " OR ".join([f"fma_source_path LIKE '{get_flexible_path_pattern(name)}'" for name in interview_names])
            queries.append(f"""
                DELETE FROM ffprobe_metadata_audio
                WHERE {conditions};
            """)
        elif tbl == "ffprobe_metadata_video":
            # Path-based PK: fmv_source_path contains interview name variants
            conditions = " OR ".join([f"fmv_source_path LIKE '{get_flexible_path_pattern(name)}'" for name in interview_names])
            queries.append(f"""
                DELETE FROM ffprobe_metadata_video
                WHERE {conditions};
            """)
        elif tbl == "ffprobe_metadata":
            # Path-based PK: fm_source_path contains interview name variants
            conditions = " OR ".join([f"fm_source_path LIKE '{get_flexible_path_pattern(name)}'" for name in interview_names])
            queries.append(f"""
                DELETE FROM ffprobe_metadata
                WHERE {conditions};
            """)
        elif tbl == "openface":
            # openface.vs_path references video_streams.vs_path
            # Follow the FK chain: video_streams -> decrypted_files -> interview_files -> interview_parts
            queries.append(f"""
                DELETE FROM openface
                WHERE vs_path IN (
                    SELECT vs_path FROM video_streams
                    WHERE video_path IN (
                        SELECT destination_path FROM decrypted_files
                        WHERE source_path IN (
                            SELECT interview_file FROM interview_files
                            INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                            WHERE interview_parts.interview_name IN ('{interview_list}')
                        )
                    )
                );
            """)
        elif tbl == "openface_qc":
            # openface_qc.of_processed_path references openface.of_processed_path
            # Follow the FK chain through openface
            queries.append(f"""
                DELETE FROM openface_qc
                WHERE of_processed_path IN (
                    SELECT of_processed_path FROM openface
                    WHERE vs_path IN (
                        SELECT vs_path FROM video_streams
                        WHERE video_path IN (
                            SELECT destination_path FROM decrypted_files
                            WHERE source_path IN (
                                SELECT interview_file FROM interview_files
                                INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
                                WHERE interview_parts.interview_name IN ('{interview_list}')
                            )
                        )
                    )
                );
            """)
        else:
            # Default: Try interview_name column
            queries.append(f"""
                DELETE FROM {tbl}
                WHERE interview_name IN ('{interview_list}');
            """)

    if dry_run:
        logger.info("[bold yellow]DRY RUN: Would execute the following queries:[/bold yellow]", extra={"markup": True})
        for i, query in enumerate(queries, 1):
            logger.info(f"\n[cyan]Query {i}:[/cyan]", extra={"markup": True})
            logger.info(query.strip())
        logger.info("\n[bold yellow]DRY RUN: No changes made to database[/bold yellow]", extra={"markup": True})
    else:
        logger.info("Executing deletions...")
        success_count = 0
        skip_count = 0
        for i, query in enumerate(queries):
            try:
                # Pass on_failure=None to raise exception instead of sys.exit(1)
                db.execute_queries(config_file=config_file, queries=[query], show_commands=False, on_failure=None)
                success_count += 1
            except Exception as e:
                error_msg = str(e).lower()
                # Skip non-existent tables gracefully
                if "does not exist" in error_msg or "relation" in error_msg:
                    # Extract table name from query for logging
                    table_match = query.strip().split("FROM")[1].split()[0] if "FROM" in query else "unknown"
                    logger.warning(f"Skipping non-existent table: {table_match}")
                    skip_count += 1
                else:
                    logger.error(f"Error executing query: {e}")
                    raise
        logger.info(f"[bold green]✓ Database records deleted successfully! ({success_count} queries executed, {skip_count} skipped)[/bold green]", extra={"markup": True})


def find_processed_folders(data_root: Path, interview_names: list):
    """
    Find all processed folders for the interviews.

    Handles both structures:
    - data_root/PROTECTED/STUDY/SUBJECT/TYPE/processed/INTERVIEW_NAME/
    - temp_root/PHOENIX/PROTECTED/STUDY/SUBJECT/TYPE/processed/INTERVIEW_NAME/
    """

    folders = []

    # Determine base paths to search (handles both PROTECTED and PHOENIX/PROTECTED structures)
    base_paths = []
    if (data_root / "PROTECTED").exists():
        base_paths.append(data_root / "PROTECTED")
    if (data_root / "PHOENIX" / "PROTECTED").exists():
        base_paths.append(data_root / "PHOENIX" / "PROTECTED")

    for interview_name in interview_names:
        # Parse interview name to build path
        info = normalize_interview_name(interview_name)

        if info["format"] in ["dpdash", "raw"]:
            subject_id = info["subject_id"]
            interview_type = info["interview_type"]

            # Common patterns
            type_variants = [
                interview_type.lower().replace("interview", "_interview"),
                interview_type.lower(),
                f"{interview_type.lower()}_interview",
            ]

            for base_path in base_paths:
                for type_var in type_variants:
                    # Pattern: BASE/STUDY/SUBJECT/TYPE/processed/INTERVIEW_NAME/
                    matches = list(base_path.glob(f"*/{subject_id}/{type_var}/processed/{interview_name}*"))
                    folders.extend(matches)

        # Also try direct glob search in entire data_root
        direct_matches = list(data_root.glob(f"**/{interview_name}*"))
        for match in direct_matches:
            if "processed" in str(match):
                folders.append(match)

    # Deduplicate
    folders = list(set(folders))
    folders.sort()

    return folders


def show_filesystem_preview(folders: list):
    """
    Show what will be deleted from filesystem.
    """

    if not folders:
        console.print("[yellow]No processed folders found.[/yellow]")
        return 0

    table = Table(title="Processed Folders to Delete")
    table.add_column("Path", style="cyan")
    table.add_column("Size", justify="right", style="yellow")

    total_size = 0

    for folder in folders:
        if folder.exists():
            size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
            size_mb = size / (1024 * 1024)
            total_size += size

            table.add_row(str(folder), f"{size_mb:.2f} MB")

    console.print(table)
    console.print(f"\n[bold yellow]Total size: {total_size / (1024 * 1024):.2f} MB[/bold yellow]")

    return len(folders)


def delete_filesystem_folders(folders: list, dry_run: bool = False):
    """
    Delete folders from filesystem.
    """

    if dry_run:
        logger.info("[bold yellow]DRY RUN: Would delete the following folders:[/bold yellow]", extra={"markup": True})
        for folder in folders:
            if folder.exists():
                logger.info(f"  • {folder}")
        logger.info("\n[bold yellow]DRY RUN: No files deleted[/bold yellow]", extra={"markup": True})
    else:
        for folder in folders:
            if folder.exists():
                logger.info(f"Deleting: {folder}")
                shutil.rmtree(folder)

        logger.info("[bold green]✓ Folders deleted successfully![/bold green]", extra={"markup": True})


def main():
    parser = argparse.ArgumentParser(
        prog=MODULE_NAME,
        description="Interactive tool to clean up interview data for reprocessing."
    )
    parser.add_argument("-c", "--config", type=str, required=False, help="Path to config file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be deleted without actually deleting")

    args = parser.parse_args()
    dry_run = args.dry_run

    # Get config
    if args.config:
        config_file = Path(args.config).resolve()
    else:
        config_file = utils.get_config_file_path()

    if not config_file.exists():
        logger.error(f"Config not found: {config_file}")
        sys.exit(1)

    console.rule("[bold red]Interview Cleanup Tool[/bold red]")
    if dry_run:
        logger.info("[bold yellow]🔍 DRY RUN MODE - No changes will be made[/bold yellow]", extra={"markup": True})
    logger.info(f"Using config: {config_file}")

    # Step 1: Get interview name
    interview_name = Prompt.ask("\n[bold cyan]Enter interview name[/bold cyan] (raw or dpdash format)")

    # Find related interviews
    logger.info(f"\nSearching for interviews matching: {interview_name}")
    related_interviews = find_related_interviews(config_file, interview_name)

    if not related_interviews:
        logger.error(f"No interviews found matching: {interview_name}")
        sys.exit(1)

    console.print(f"\n[bold green]Found {len(related_interviews)} interview(s):[/bold green]")
    for interview in related_interviews:
        console.print(f"  • {interview}")

    # Step 2: Choose what to delete from database
    console.rule("[bold blue]Database Cleanup[/bold blue]")

    delete_choice = Prompt.ask(
        "\n[bold cyan]What to delete from database?[/bold cyan]",
        choices=["all", "specific", "skip"],
        default="all"
    )

    if delete_choice == "skip":
        logger.info("Skipping database cleanup")
    elif delete_choice == "specific":
        table_name = Prompt.ask("[bold cyan]Enter table name[/bold cyan]")
        console.print(f"\n[bold yellow]Preview: Deleting from {table_name}[/bold yellow]")
        count = show_database_preview(config_file, related_interviews, table_name)

        if count > 0:
            if dry_run:
                delete_database_records(config_file, related_interviews, table_name, dry_run=True)
            else:
                confirmation = Prompt.ask("\n[bold red]Type 'delete' to confirm[/bold red]")
                if confirmation.lower() == "delete":
                    delete_database_records(config_file, related_interviews, table_name, dry_run=False)
                else:
                    logger.info("Cancelled.")
    else:  # all
        console.print("\n[bold yellow]Preview: All database tables[/bold yellow]")
        count = show_database_preview(config_file, related_interviews)

        if count > 0:
            if dry_run:
                delete_database_records(config_file, related_interviews, dry_run=True)
            else:
                confirmation = Prompt.ask("\n[bold red]Type 'delete' to confirm[/bold red]")
                if confirmation.lower() == "delete":
                    delete_database_records(config_file, related_interviews, dry_run=False)
                else:
                    logger.info("Cancelled.")

    # Step 3: Delete from processed folders
    console.rule("[bold blue]Filesystem Cleanup[/bold blue]")

    delete_fs = Confirm.ask("\n[bold cyan]Delete processed folders?[/bold cyan]", default=False)

    if delete_fs:
        data_root = orchestrator.get_data_root(config_file=config_file, enforce_real=True)

        # Search in both data_root and temp_root
        search_paths = [data_root]
        if TEMP_ROOT.exists():
            search_paths.append(TEMP_ROOT)

        all_folders = []
        for search_path in search_paths:
            logger.info(f"Searching in: {search_path}")
            folders = find_processed_folders(search_path, related_interviews)
            all_folders.extend(folders)

        # Deduplicate
        all_folders = list(set(all_folders))
        all_folders.sort()

        if all_folders:
            console.print("\n[bold yellow]Preview: Processed folders[/bold yellow]")
            count = show_filesystem_preview(all_folders)

            if count > 0:
                if dry_run:
                    delete_filesystem_folders(all_folders, dry_run=True)
                else:
                    confirmation = Prompt.ask("\n[bold red]Type 'delete' to confirm[/bold red]")
                    if confirmation.lower() == "delete":
                        delete_filesystem_folders(all_folders, dry_run=False)
                    else:
                        logger.info("Cancelled.")
        else:
            logger.info("No processed folders found.")

    console.rule("[bold green]Cleanup Complete[/bold green]")
    logger.info("Interview cleanup finished!")


if __name__ == "__main__":
    main()
