import re
import json
from pathlib import Path
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def transcript_to_df(transcript_path: Path) -> pd.DataFrame:
    """
    Reads a transcript JSON file saved as txt and returns a dataframe with the following columns:
    - start_time
    - end_time
    - speaker
    - text
    Expected JSON format:
    ```
    {
        "speaker_turns": "Patient [142.440 - 193.390]\nHi, how?\n\nClinician [193.390 - 203.240]\nHi, how are you?",
        "timestamped_transcription": "..."
    }
    ```
    Speaker/time pattern to match either:
    - Patient [SSS.MMM - SSS.MMM]
    - Clinician [SSS.MMM - SSS.MMM]
    - <speaker:N> [SSS.MMM - SSS.MMM] where N is 1-2 digits
    Timestamp format:
    - seconds.milliseconds (e.g., 142.440 - 193.390)
    Args:
        transcript_path: Path, path to the transcript JSON file
    Returns:
        pd.DataFrame
    """

    chunks = []

    # Load JSON and extract speaker_turns string
    with open(str(transcript_path), "r", encoding="utf-8") as f:
        data = json.load(f)
    
    speaker_turns = data.get("speaker_turns", "")
    
    if not speaker_turns:
        logger.warning("No 'speaker_turns' key found in JSON or it's empty")
        return pd.DataFrame(columns=["start_time", "end_time", "speaker", "text"])
    
    # Split into lines
    lines = speaker_turns.split('\n')

    i = 0
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue

        # Pattern to match either:
        # - Patient [SSS.MMM - SSS.MMM]
        # - Clinician [SSS.MMM - SSS.MMM]
        # - <speaker:N> [SSS.MMM - SSS.MMM] where N is 1-2 digits
        # Format: seconds.milliseconds (e.g., 142.440 - 193.390)
        pattern = r'^(Patient|Clinician|<speaker:\d{1,2}>)\s*\[(\d+\.\d+)\s*-\s*(\d+\.\d+)\]'
        match = re.match(pattern, line)
       
        if match:
            speaker = match.group(1)
            start_time = match.group(2)
            end_time = match.group(3)
            
            # The next line(s) should contain the text
            text_lines = []
            i += 1
            
            # Collect all text lines until we hit another speaker line or empty line
            while i < len(lines):
                next_line = lines[i].strip()
                
                # Check if this is another speaker line
                if re.match(pattern, next_line):
                    break
                
                # Add non-empty lines to text
                if next_line:
                    text_lines.append(next_line)
                    i += 1
                else:
                    # Empty line - move past it but stop collecting text
                    i += 1
                    break
            
            # Combine all text lines
            text = " ".join(text_lines)
            
            if text:  # Only add if there's actual text
                chunks.append({
                    "start_time": start_time,
                    "end_time": end_time,
                    "speaker": speaker,
                    "text": text.strip(),
                })
            else:
                logger.warning(f"Speaker '{speaker}' at {start_time} has no text")
        else:
            logger.warning(f"Skipped parsing line: '{line}'")
            i += 1

    df = pd.DataFrame(chunks)
    
    return df

my_df = transcript_to_df('/mnt/PREDiCTOR/AV_TEMP/redac_temp/transcription_outputs/PREDiCTOR-P0014HA-onsiteInterview_MD-day0002_dpi_transcript.txt')
