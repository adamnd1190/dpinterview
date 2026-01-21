# diarization script using pyannote.audio

import os
import pathlib
import torch
import argparse
from pathlib import Path
import numpy as np


from utils import (
    load_audio_from_file,
    find_all_speech_segments,
    get_onsite_files,
    get_offsite_files,
    get_audio_files,
    get_cognition_files,
    get_participant_id,
    setup_diarization_pipeline
)

def diarize_audio(audio, sr, pipeline, min_speakers=1, max_speakers=6):
    """
    Perform speaker diarization on audio.
    
    Args:
        audio (np.array): Audio data
        sr (int): Sample rate
        pipeline: Diarization pipeline
        min_speakers (int, optional): Minimum number of speakers
        max_speakers (int, optional): Maximum number of speakers
        
    Returns:
        Diarization: Diarization result
    """
    # Build kwargs for pipeline call
    kwargs = {}
    if min_speakers is not None:
        kwargs['min_speakers'] = min_speakers
    if max_speakers is not None:
        kwargs['max_speakers'] = max_speakers
    
    return pipeline(
        {"waveform": torch.tensor(audio[None, :]), "sample_rate": sr},
        **kwargs
    )

def create_speech_mask(audio, segments, sr):
    """
    Create a binary mask for speech regions and apply it to audio.
    
    Args:
        audio (np.array): Original audio
        segments (list): List of (start_time, end_time, segment_audio) tuples
        sr (int): Sample rate
        
    Returns:
        np.array: Masked audio (silence in non-speech regions)
    """
    # Create a copy of audio filled with silence
    masked_audio = np.zeros_like(audio)
    
    # Fill in speech segments
    for start_time, end_time, _ in segments:
        start_sample = int(start_time * sr)
        end_sample = int(end_time * sr)
        masked_audio[start_sample:end_sample] = audio[start_sample:end_sample]
    
    return masked_audio

def process_files(parent_paths, 
                  output_dir, 
                  token, 
                  sr=16000, 
                  device='cuda',
                  audio_diary=False,
                  onsite_interview=False,
                  offsite_interview=False,
                  cognition=False):
    """
    Process all video files for diarization.
    
    Args:
        parent_paths (list): List of input directories
        output_dir (str): Output directory
        token (str): Hugging Face token
        sr (int): Sample rate
        device (str): Device to use
    """
    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up pipeline
    pipeline = setup_diarization_pipeline(token, device)
    collar = 180
    
    # Get all files
    if onsite_interview:
        files = get_onsite_files(parent_paths)
    elif offsite_interview:
        files = get_offsite_files(parent_paths)
    elif audio_diary:
        files = get_audio_files(parent_paths)
    elif cognition:
        files = get_cognition_files(parent_paths)
        collar = 3600
    else:
        raise Exception('Must specify type of encounter to transcribe')
    
    for file_path in files:
        participant = get_participant_id(file_path)
        print(f"Processing diarization for: {participant}")
        print(f"File: {file_path}")
        
        try:
            rttm_output_path = os.path.join(output_dir, f'{participant}_diarization.rttm')
            my_file = Path(rttm_output_path)
            if my_file.is_file():
                print('Diarization RTTM exists, skipping')
                continue
            # Load audio
            print('Loading audio')
            audio, _ = load_audio_from_file(file_path, sr)

            # Find all speech segments
            print('Finding all speech segments')
            segments = find_all_speech_segments(audio, sr, collar=collar)
            print(f"Found {len(segments)} segment(s)")

            # Create masked audio for diarization (silence non-speech regions)
            print('Creating speech mask for diarization')
            masked_audio = create_speech_mask(audio, segments, sr)

            # Perform diarization on masked audio
            print("Performing diarization")
            diarization = diarize_audio(masked_audio, sr, pipeline, min_speakers=1, max_speakers=6)
            
            # Save diarization result
            rttm_output_path = os.path.join(output_dir, f'{participant}_diarization.rttm')
            with open(rttm_output_path, 'w') as f:
                diarization.write_rttm(f)
            
            print(f"Saved diarization: {rttm_output_path}")
            print("---")
            
        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            continue


def main():
    parser = argparse.ArgumentParser(description='Perform speaker diarization on video files')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='Input directories containing video files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--token', required=True,
                        help='Hugging Face token for pyannote access')
    parser.add_argument('--sample_rate', type=int, default=16000,
                        help='Audio sample rate (default: 16000)')
    parser.add_argument('--device', default='cuda',
                        help='Device to use (cuda/cpu, default: cuda)')
    parser.add_argument('--audio_diary', action='store_true')
    parser.add_argument('--onsite_interview', action='store_true')
    parser.add_argument('--offsite_interview', action='store_true')
    parser.add_argument('--cognition', action='store_true')
    
    args = parser.parse_args()
    
    process_files(
        parent_paths=args.input_dirs,
        output_dir=args.output_dir,
        token=args.token,
        sr=args.sample_rate,
        device=args.device,
        audio_diary=args.audio_diary,
        onsite_interview=args.onsite_interview,
        offsite_interview=args.offsite_interview,
        cognition=args.cognition
    )


if __name__ == "__main__":
    main()