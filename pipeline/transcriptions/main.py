#!/usr/bin/env python3
"""
Main pipeline script to run transcription, diarization, and speaker alignment.
"""

import argparse
import subprocess
import sys
from pathlib import Path
from pipeline.helpers import db, cli, notifications
from pipeline.helpers.config import config


def run_command(cmd, description):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"Starting: {description}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✓ Completed: {description}")
        return result
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Error in {description}: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Run transcription, diarization, and speaker alignment pipeline'
    )
    
    # Common arguments
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='Input directories containing video files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    
    # Transcription arguments
    parser.add_argument('--model_size', default='medium',
                        help='Whisper model size (default: medium)')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (default: 16000)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')
    
    # File type flags
    parser.add_argument('--audio_diary', action='store_true')
    parser.add_argument('--onsite_interview', action='store_true')
    parser.add_argument('--offsite_interview', action='store_true')
    parser.add_argument('--cognition', action='store_true')
    
    # Pipeline control
    parser.add_argument('--skip_transcribe', action='store_true',
                        help='Skip transcription step')
    parser.add_argument('--skip_diarize', action='store_true',
                        help='Skip diarization step')
    parser.add_argument('--skip_alignment', action='store_true',
                        help='Skip speaker alignment step')
    
    # Virtual environment paths
    parser.add_argument('--diarize_venv', default='/home/linlab/diarization_venv',
                        help='Path to diarization virtual environment')
    parser.add_argument('--cpp_venv', default='/home/linlab/cpp_venv',
                        help='Path to CPP virtual environment')
    
    args = parser.parse_args()
    
    # Get script directory
    script_dir = Path(__file__).parent
    
    # Prepare common arguments
    input_dirs_args = ['--input_dirs'] + args.input_dirs
    output_dir_arg = ['--output_dir', args.output_dir]
    
    # File type flags
    type_flags = []
    if args.audio_diary:
        type_flags.append('--audio_diary')
    if args.onsite_interview:
        type_flags.append('--onsite_interview')
    if args.offsite_interview:
        type_flags.append('--offsite_interview')
    if args.cognition:
        type_flags.append('--cognition')
    
    # Step 1: Transcription
    if not args.skip_transcribe:
        transcribe_cmd = [
            f"{args.diarize_venv}/bin/python",
            str(script_dir / "transcribe.py"),
            *input_dirs_args,
            *output_dir_arg,
            '--model_size', args.model_size,
            '--sample_rate', str(args.sample_rate),
            '--device', args.device,
            *type_flags
        ]
        run_command(transcribe_cmd, "Transcription")
    else:
        print("Skipping transcription step")
    
    # Step 2: Diarization
    if not args.skip_diarize:
        diarize_cmd = [
            f"{args.diarize_venv}/bin/python",
            str(script_dir / "diarize.py"),
            *input_dirs_args,
            *output_dir_arg,
            *type_flags
        ]
        run_command(diarize_cmd, "Diarization")
    else:
        print("Skipping diarization step")
    
    # Step 3: Speaker Alignment
    if not args.skip_alignment:
        alignment_cmd = [
            f"{args.cpp_venv}/bin/python",
            str(script_dir / "make_speaker_turns.py"),
            *input_dirs_args,
            *output_dir_arg,
            *type_flags
        ]
        run_command(alignment_cmd, "Speaker Alignment")
    else:
        print("Skipping speaker alignment step")
    
    print(f"\n{'='*60}")
    print("Pipeline completed successfully!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()