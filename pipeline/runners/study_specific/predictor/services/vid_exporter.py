#!/usr/bin/env python
"""
Early Video Exporter - Exports downscaled videos immediately after import

This runs AFTER the importer and BEFORE the rest of the pipeline.
It copies the downscaled video to the processed folder on NAS so the 
dpinterview-web dashboard can access it immediately.

The main exporter (at the end of the pipeline) will skip re-exporting
videos that were already exported by this script.
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
from typing import Optional
from datetime import datetime

from rich.logging import RichHandler

from pipeline import orchestrator
from pipeline.helpers import cli, utils, dpdash, db

MODULE_NAME = "predictor-exporter"  # Use existing exporter logging config

logger = logging.getLogger(MODULE_NAME)
logargs = {
    "level": logging.DEBUG,
    "format": "%(message)s",
    "handlers": [RichHandler(rich_tracebacks=True)],
}
logging.basicConfig(**logargs)

console = utils.get_console()


def get_interview_to_export(config_file: Path, study_id: str) -> Optional[str]:
    """
    Get an interview that has been imported but not yet had its video exported early.
    Only returns interviews where the decrypted video file still exists.
    
    Args:
        config_file: Path to config file
        study_id: Study ID
        
    Returns:
        interview_name or None
    """
    # REVERT NOTE (2026-01-05): Removed requested_by filter to include importer-decrypted files.
    # Original had: AND decrypted_files.requested_by = 'fetch_video'
    # This excluded PREDICTOR videos decrypted by the importer. To revert, add that line back.
    #
    # Also added check to ensure main exporter hasn't processed the interview yet
    # (by checking for 'streams' export record). Main exporter deletes temp files after exporting.
    query = f"""
        SELECT interviews.interview_name, decrypted_files.destination_path
        FROM interviews
        INNER JOIN interview_parts ON interviews.interview_name = interview_parts.interview_name
        INNER JOIN interview_files ON interview_parts.interview_path = interview_files.interview_path
        INNER JOIN decrypted_files ON interview_files.interview_file = decrypted_files.source_path
        WHERE interviews.study_id = '{study_id}'
            AND decrypted_files.decrypted = true
            AND interviews.interview_name NOT IN (
                SELECT interview_name
                FROM exported_assets
                WHERE asset_tag = 'early_video_export'
            )
            AND interviews.interview_name NOT IN (
                SELECT interview_name
                FROM exported_assets
                WHERE asset_tag = 'streams'
            )
        ORDER BY decrypted_files.decrypted_at ASC
        LIMIT 20;
    """
    
    df = db.execute_sql(config_file=config_file, query=query)
    
    if df.empty:
        return None
    
    # Check which videos actually exist on disk
    for _, row in df.iterrows():
        interview_name = row['interview_name']
        video_path = Path(row['destination_path'])
        
        if video_path.exists():
            return interview_name
        else:
            logger.debug(f"Skipping {interview_name} - video file no longer exists: {video_path}")
    
    return None


def get_downscaled_video_path(config_file: Path, interview_name: str) -> Optional[Path]:
    """
    Get the path to the downscaled video for an interview.

    Args:
        config_file: Path to config file
        interview_name: Interview name

    Returns:
        Path to downscaled video or None

    REVERT NOTE (2026-01-05): Removed requested_by filter to work with importer-decrypted files.
    Original had: AND decrypted_files.requested_by = 'fetch_video'
    To revert, add that line back at line 129.
    """
    query = f"""
        SELECT destination_path
        FROM decrypted_files
        INNER JOIN interview_files ON decrypted_files.source_path = interview_files.interview_file
        INNER JOIN interview_parts ON interview_files.interview_path = interview_parts.interview_path
        INNER JOIN interviews ON interview_parts.interview_name = interviews.interview_name
        WHERE interviews.interview_name = '{interview_name}'
            AND decrypted_files.decrypted = true
        LIMIT 1;
    """

    result = db.fetch_record(config_file=config_file, query=query)

    if result:
        return Path(result)
    return None


def construct_video_export_path(interview_name: str, config_file: Path) -> Path:
    """
    Constructs export path for downscaled video.
    
    Structure: {subject_id}/{interview_type}/processed/{interview_name}/video/
    
    Args:
        interview_name: Interview name (e.g., PREDiCTOR-P0001MA-offsiteInterview_CRC_Baseline-day0009)
        config_file: Path to config file
        
    Returns:
        Export path
    """
    data_root = orchestrator.get_data_root(config_file=config_file, enforce_real=True)
    
    dpdash_dict = dpdash.parse_dpdash_name(interview_name)
    study = dpdash_dict["study"]
    subject_id = dpdash_dict["subject"]
    
    # Extract base interview type (ignore CRC/MD tags)
    # "offsiteInterview_CRC_Baseline" → "offsiteInterview"
    data_type_camel = dpdash_dict["data_type"]
    base_data_type = data_type_camel.split('_')[0]
    
    # Convert to snake_case
    interview_type = ''.join(['_' + c.lower() if c.isupper() else c for c in base_data_type]).lstrip('_')
    
    export_path = (
        data_root
        / "PROTECTED"
        / study
        / subject_id
        / interview_type  # Just "offsite_interview" or "onsite_interview"
        / "processed"
        / interview_name  # Full name: "PREDiCTOR-P0001MA-offsiteInterview_CRC_Baseline-day0009"
        / "video"
    )
    
    return export_path


def export_video_early(config_file: Path, interview_name: str, debug: bool = False) -> bool:
    """
    Export downscaled video to processed folder immediately.
    If video is > 5GB, re-encode to 720p for dashboard compatibility.
    
    Args:
        config_file: Path to config file
        interview_name: Interview name
        debug: If True, don't actually copy files
        
    Returns:
        True if successful, False otherwise
    """
    # Get downscaled video path
    video_path = get_downscaled_video_path(config_file=config_file, interview_name=interview_name)
    
    if not video_path:
        logger.warning(f"No downscaled video found for {interview_name}")
        # Mark as processed anyway to avoid retry loop
        if not debug:
            from pipeline.models.exported_assets import ExportedAsset
            asset = ExportedAsset(
                interview_name=interview_name,
                asset_path=Path("/dev/null"),  # Placeholder
                asset_type="file",
                asset_export_type="PROTECTED",
                asset_tag="early_video_export",
                asset_destination=Path("/dev/null"),  # Placeholder
                aset_exported_timestamp=datetime.now(),
            )
            query = asset.to_sql()
            db.execute_queries(config_file=config_file, queries=[query])
            logger.info(f"Marked {interview_name} as processed (video not found)")
        return False
    
    logger.info(f"Found downscaled video: {video_path}")
    
    if not video_path.exists():
        logger.warning(f"Video file does not exist (likely already cleaned up): {video_path}")
        # Mark as processed anyway to avoid retry loop
        if not debug:
            from pipeline.models.exported_assets import ExportedAsset
            asset = ExportedAsset(
                interview_name=interview_name,
                asset_path=video_path,
                asset_type="file",
                asset_export_type="PROTECTED",
                asset_tag="early_video_export",
                asset_destination=Path("/dev/null"),  # Placeholder
                aset_exported_timestamp=datetime.now(),
            )
            query = asset.to_sql()
            db.execute_queries(config_file=config_file, queries=[query])
            logger.info(f"Marked {interview_name} as processed (video already deleted)")
        return False
    
    file_size_mb = video_path.stat().st_size / (1024*1024)
    file_size_gb = file_size_mb / 1024
    logger.info(f"Video file size: {file_size_gb:.2f} GB")
    
    # Check if we need to re-encode for dashboard
    MAX_SIZE_GB = 5.0
    if file_size_gb > MAX_SIZE_GB:
        logger.warning(f"Video exceeds {MAX_SIZE_GB}GB, re-encoding to 720p for dashboard compatibility...")
        
        # Construct destination path first
        export_path = construct_video_export_path(interview_name=interview_name, config_file=config_file)
        destination_path = export_path / video_path.name
        
        # Create temporary 720p version with original filename
        temp_720p_path = video_path.parent / f"temp_720p_{video_path.name}"
        
        if not debug:
            import subprocess
            
            # FFmpeg command: 720p, higher compression (CRF 28), fast encoding
            ffmpeg_cmd = [
                "ffmpeg",
                "-i", str(video_path),
                "-vf", "scale=-2:720",  # Scale to 720p, maintain aspect ratio
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "28",  # Higher CRF = more compression (23 is default, 28 is ~50% smaller)
                "-c:a", "aac",
                "-b:a", "128k",
                "-movflags", "+faststart",
                "-y",  # Overwrite if exists
                str(temp_720p_path)
            ]
            
            logger.info(f"Running: {' '.join(ffmpeg_cmd)}")
            
            try:
                result = subprocess.run(
                    ffmpeg_cmd,
                    capture_output=True,
                    text=True,
                    timeout=3600  # 1 hour timeout
                )
                
                if result.returncode != 0:
                    logger.error(f"FFmpeg failed: {result.stderr}")
                    return False
                
                if not temp_720p_path.exists():
                    logger.error(f"720p video was not created: {temp_720p_path}")
                    return False
                
                new_size_gb = temp_720p_path.stat().st_size / (1024*1024*1024)
                reduction = ((file_size_gb - new_size_gb) / file_size_gb) * 100
                logger.info(f"Re-encoded to 720p: {new_size_gb:.2f} GB ({reduction:.1f}% reduction)")
                
                # Use the 720p version for export (but keep original filename)
                video_path = temp_720p_path
                
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg encoding timeout (>1 hour)")
                return False
            except Exception as e:
                logger.error(f"Error during re-encoding: {str(e)}")
                return False
        else:
            logger.info(f"[DEBUG MODE] Would re-encode to 720p")
            # In debug mode, still set destination_path for logging
            export_path = construct_video_export_path(interview_name=interview_name, config_file=config_file)
            destination_path = export_path / video_path.name
    else:
        # Normal case: construct destination path
        export_path = construct_video_export_path(interview_name=interview_name, config_file=config_file)
        destination_path = export_path / video_path.name
    
    logger.info(f"Exporting video early for dashboard: {interview_name}")
    logger.info(f"Source: {video_path}")
    logger.info(f"Destination: {destination_path}")
    
    if not debug:
        try:
            # Create destination directory
            logger.info(f"Creating directory: {destination_path.parent}")
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Verify directory was created
            if not destination_path.parent.exists():
                logger.error(f"Failed to create directory: {destination_path.parent}")
                return False
            
            logger.info(f"Directory created successfully")
            
            # Copy file
            logger.info(f"Copying file...")
            cli.copy(video_path, destination_path)
            
            # Verify copy
            if not destination_path.exists():
                logger.error(f"File was not copied to destination: {destination_path}")
                return False
            
            dest_size = destination_path.stat().st_size / (1024*1024)
            logger.info(f"File copied successfully, size: {dest_size:.2f} MB")
            
            # Clean up temp 720p file if we created one
            if video_path.name.startswith("temp_720p_"):
                logger.info(f"Cleaning up temporary 720p file: {video_path}")
                video_path.unlink()
            
            # Record export in database
            from pipeline.models.exported_assets import ExportedAsset

            # REVERT NOTE (2026-01-05): Changed asset_path from video_path to destination_path
            # to record where the file was exported TO, not FROM. Web dashboard uses this to find videos.
            # To revert: change asset_path=destination_path back to asset_path=video_path
            asset = ExportedAsset(
                interview_name=interview_name,
                asset_path=destination_path,  # Changed from video_path - records export destination
                asset_type="file",
                asset_export_type="PROTECTED",
                asset_tag="early_video_export",
                asset_destination=destination_path,
                aset_exported_timestamp=datetime.now(),
            )
            
            query = asset.to_sql()
            db.execute_queries(config_file=config_file, queries=[query])
            
            logger.info(f"✓ Exported video for {interview_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error during export: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    else:
        logger.info(f"[DEBUG MODE] Would export video for {interview_name}")
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="early-video-exporter",
        description="Export downscaled videos immediately after import for dashboard access."
    )
    parser.add_argument(
        "-c", "--config", type=str, help="Path to the config file.", required=False
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (don't actually copy files).",
        default=False,
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

    debug: bool = args.debug

    utils.configure_logging(
        config_file=config_file, module_name=MODULE_NAME, logger=logger
    )

    console.rule(f"[bold red]{MODULE_NAME}")
    logger.info(f"Using config file: {config_file}")

    if debug:
        logger.warning("DEBUG MODE: No files will be copied")

    studies = orchestrator.get_studies(config_file=config_file)
    COUNTER = 0

    logger.info(f"Monitoring {len(studies)} studies for videos to export")

    while True:
        exported_any = False
        
        for study_id in studies:
            interview_name = get_interview_to_export(config_file=config_file, study_id=study_id)
            
            if interview_name:
                success = export_video_early(
                    config_file=config_file,
                    interview_name=interview_name,
                    debug=debug
                )
                
                if success:
                    COUNTER += 1
                    exported_any = True

        if not exported_any:
            # No videos to export, snooze
            if COUNTER > 0:
                orchestrator.log(
                    config_file=config_file,
                    module_name=MODULE_NAME,
                    message=f"Exported {COUNTER} videos for dashboard.",
                )
                COUNTER = 0
            
            orchestrator.snooze(config_file=config_file)