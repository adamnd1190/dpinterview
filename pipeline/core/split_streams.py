"""
Helper functions to split video into streams.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

from pipeline import orchestrator
from pipeline.helpers import db, dpdash, ffmpeg, utils
from pipeline.helpers.timer import Timer
from pipeline.models.interview_roles import InterviewRole
from pipeline.models.video_streams import VideoStream

logger = logging.getLogger(__name__)


def get_file_to_process(
    config_file: Path, study_id: str
) -> Optional[Tuple[Path, bool, int, int, int]]:
    """
    Fetch a file to process from the database.

    - Fetches a file that has not been processed yet and is part of the study.
        - Must have metadata extracted

    Args:
        config_file (Path): Path to config file

    Returns:
        Tuple of (video_path, has_black_bars, black_bar_height, width, height) or None
    """
    sql_query = f"""
        SELECT vqqc.video_path, vqqc.has_black_bars, vqqc.black_bar_height,
               fmv.fmv_width, fmv.fmv_height
        FROM video_quick_qc AS vqqc
        LEFT JOIN ffprobe_metadata_video AS fmv ON vqqc.video_path = fmv.fmv_source_path
        LEFT JOIN (
            SELECT decrypted_files.destination_path, interview_files.interview_file_tags
            FROM interview_files
            LEFT JOIN decrypted_files ON interview_files.interview_file = decrypted_files.source_path
        ) AS if
            ON vqqc.video_path = if.destination_path
        WHERE vqqc.video_path NOT IN (
            SELECT video_path
            FROM video_streams
        ) AND vqqc.video_path IN (
            SELECT destination_path
            FROM decrypted_files
            LEFT JOIN interview_files ON interview_files.interview_file = decrypted_files.source_path
            LEFT JOIN interview_parts USING (interview_path)
            LEFT JOIN interviews USING (interview_name)
            WHERE interviews.study_id = '{study_id}'
        )
        ORDER BY RANDOM()
        LIMIT 1;
    """

    result_df = db.execute_sql(config_file=config_file, query=sql_query)
    if result_df.empty:
        return None

    video_path = Path(result_df.iloc[0]["video_path"])
    has_black_bars = bool(result_df.iloc[0]["has_black_bars"])
    black_bar_height = result_df.iloc[0]["black_bar_height"]
    if black_bar_height is not None:
        black_bar_height = int(black_bar_height)

    # Get video dimensions
    width = result_df.iloc[0]["fmv_width"]
    height = result_df.iloc[0]["fmv_height"]
    if width is not None:
        width = int(width)
    if height is not None:
        height = int(height)

    return video_path, has_black_bars, black_bar_height, width, height


def construct_stream_path(video_path: Path, role: InterviewRole, suffix: str) -> Path:
    """
    Constructs a dpdash compliant stream path

    Args:
        video_path (Path): Path to video
        role (InterviewRole): Role of the stream
        suffix (str): Suffix of the stream
    """
    dpdash_dict = dpdash.parse_dpdash_name(video_path.name)
    if dpdash_dict["optional_tags"] is None:
        optional_tag: List[str] = []
    else:
        optional_tag = dpdash_dict["optional_tags"]  # type: ignore

    optional_tag.append(role.value)
    dpdash_dict["optional_tags"] = optional_tag

    dpdash_dict["category"] = "video"

    dp_dash_name = dpdash.get_dpdash_name_from_dict(dpdash_dict)
    stream_path = video_path.parent / "streams" / f"{dp_dash_name}.{suffix}"

    # Create streams directory if it doesn't exist
    stream_path.parent.mkdir(parents=True, exist_ok=True)

    return stream_path


def split_streams(
    video_path: Path,
    has_black_bars: bool,
    black_bar_height: Optional[int],
    config_file: Path,
    width: Optional[int] = None,
    height: Optional[int] = None,
) -> List[VideoStream]:
    """
    Split video into streams

    Args:
        video_path (Path): Path to video
        has_black_bars (bool): Whether video has black bars
        black_bar_height (Optional[int]): Height of black bars
        config_file (Path): Path to config file
        width (Optional[int]): Video width from ffprobe_metadata_video
        height (Optional[int]): Video height from ffprobe_metadata_video
    """
    config_params = utils.config(path=config_file, section="split-streams")
    default_role = InterviewRole.from_str(config_params["default_role"])

    streams = []
    logger.info(f"Splitting streams for {video_path}...", extra={"markup": True})

    # Check if this is an onsite interview
    is_onsite = "onsite_interview" in str(video_path).lower()

    # Check if this is a single-side zoom video (640x720 resolution)
    # These videos have only subject side, need black bar removal but no split
    is_single_side_zoom = (width == 640 and height == 720)

    # Check if this is a dual-camera onsite video that needs splitting
    # Dual-camera format has ~32:9 aspect ratio (e.g., 3840x1080)
    # All other onsite videos are single-camera and should skip splitting
    is_dual_camera_onsite = False
    if is_onsite and width is not None and height is not None and height > 0:
        aspect_ratio = width / height
        # Dual-camera ratio is around 3.5:1 (32:9), allow some tolerance
        is_dual_camera_onsite = 3.0 <= aspect_ratio <= 4.0

    # Single-camera onsite: any onsite video that's NOT dual-camera format
    is_single_camera_onsite = is_onsite and not is_dual_camera_onsite

    if is_single_side_zoom or is_single_camera_onsite:
        if is_single_side_zoom:
            logger.info(f"Single-side zoom video detected ({width}x{height}) - creating subject video only")
        else:
            logger.info(f"Single-camera onsite video detected ({width}x{height}) - creating subject video only")
        subject_role = InterviewRole.SUBJECT

        if black_bar_height is None:
            black_bar_height = 0

        # Crop params: full width, remove black bars from top and bottom
        crop_params = f"iw:ih-{2 * black_bar_height}:0:{black_bar_height}"

        stream_file_path = construct_stream_path(
            video_path=video_path, role=subject_role, suffix="mp4"
        )

        with Timer() as timer:
            with utils.get_progress_bar() as progress:
                task = progress.add_task("Creating subject stream", total=1)
                progress.update(task, description=f"Creating {subject_role.value} stream")

                ffmpeg.crop_video(
                    source=video_path,
                    target=stream_file_path,
                    crop_params=crop_params,
                    progress=progress,
                )
                orchestrator.fix_permissions(
                    config_file=config_file, file_path=stream_file_path
                )
                progress.update(task, advance=1)

        logger.info(
            f"Created {subject_role.value} stream: {stream_file_path} ({timer.duration})",
            extra={"markup": True},
        )

        stream: VideoStream = VideoStream(
            video_path=video_path,
            ir_role=subject_role,
            vs_path=stream_file_path,
            vs_process_time=timer.duration,
        )
        streams.append(stream)
        return streams

    if not has_black_bars:
        if is_onsite:
            # OBS onsite videos: no black bars but still split
            logger.info("OBS onsite video detected - splitting without black bar removal")
            black_bar_height = 0
        else:
            # Offsite/Zoom: no black bars means no splitting
            logger.info("No black bars detected - skipping split")
            stream: VideoStream = VideoStream(
                video_path=video_path, ir_role=default_role, vs_path=video_path
            )
            streams.append(stream)
            return streams
    else:
        # Has black bars - process normally
        logger.info("Black bars detected - cropping and splitting")
        if black_bar_height is None:
            black_bar_height = 0

    # Continue with splitting (only reached if should split)
    left_crop_params = f"iw/2:ih-{2 * black_bar_height}:0:{black_bar_height}"
    right_crop_params = f"iw/2:ih-{2 * black_bar_height}:iw/2:{black_bar_height}"

    left_role = InterviewRole.from_str(config_params["left_role"])
    right_role = InterviewRole.from_str(config_params["right_role"])

    with utils.get_progress_bar() as progress:
        task = progress.add_task("Splitting streams", total=2)
        for role, crop_params in [
            (left_role, left_crop_params),
            (right_role, right_crop_params),
        ]:
            progress.update(task, description=f"Splitting {role.value} stream")
            stream_file_path = construct_stream_path(
                video_path=video_path, role=role, suffix="mp4"
            )

            with Timer() as timer:
                ffmpeg.crop_video(
                    source=video_path,
                    target=stream_file_path,
                    crop_params=crop_params,
                    progress=progress,
                )
                orchestrator.fix_permissions(
                    config_file=config_file, file_path=stream_file_path
                )

            logger.info(
                f"Split {role.value} stream: {stream_file_path} ({timer.duration})",
                extra={"markup": True},
            )

            stream: VideoStream = VideoStream(
                video_path=video_path,
                ir_role=role,
                vs_path=stream_file_path,
                vs_process_time=timer.duration,
            )
            streams.append(stream)
            progress.update(task, advance=1)

    return streams


def log_streams(config_file: Path, streams: List[VideoStream]) -> None:
    """
    Log streams to database.

    Args:
        config_file (Path): Path to config file
        streams (List[VideoStream]): List of streams
    """
    sql_queries = [stream.to_sql() for stream in streams]

    logger.info("Inserting streams into DB", extra={"markup": True})
    db.execute_queries(config_file=config_file, queries=sql_queries)
