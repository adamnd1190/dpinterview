# transcription script using Whisper
import os
import json
import pathlib
import argparse
import whisper_timestamped as whisper
from pathlib import Path


from utils import (
    load_audio_from_file,
    find_all_speech_segments,
    get_onsite_files,
    get_offsite_files,
    get_audio_files,
    get_cognition_files,
    get_participant_id,
    setup_whisper_model
)


def transcribe_audio(audio, model, initial_prompt=None):
    """
    Transcribe audio using Whisper.
    
    Args:
        audio (np.array): Audio data
        model: Whisper model
        initial_prompt (str): Initial prompt for transcription
        
    Returns:
        dict: Transcription result
    """
    return whisper.transcribe(
        model,
        audio,
        language='en',
        initial_prompt=initial_prompt,
        beam_size=10,
        vad=True,
        condition_on_previous_text=False  
    )

def process_files(parent_paths, 
                  output_dir, 
                  model_size='base', 
                  sr=16000, 
                  device='cuda', 
                  audio_diary=False,
                  onsite_interview=False,
                  offsite_interview=False,
                  cognition=False
                  ):
    """
    Process all video files for transcription.
    
    Args:
        parent_paths (list): List of input directories
        output_dir (str): Output directory
        model_size (str): Whisper model size
        sr (int): Sample rate
        device (str): Device to use
    """
    # Create output directory
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set up Whisper model
    whisper_model = setup_whisper_model(model_size, device)
    
    # Initial prompt for better transcription
    initial_prompt = "Hi, how are you doing today? Good. Good, I'm glad to hear that. Yeah, uh. One, two, three."
    
    collar = 180 #Gonna keep this in place even though we segment based on continuous segments
    # Get all files
    if onsite_interview:
        files = get_onsite_files(parent_paths)
    elif offsite_interview:
        files = get_offsite_files(parent_paths)
    elif audio_diary:
        files = get_audio_files(parent_paths)
    elif cognition:
        files = get_cognition_files(parent_paths)
        # collar = 3600
    else:
        raise Exception('Must specify type of encounter to transcribe')
    
    

    for file_path in files:
        participant = get_participant_id(file_path)
        print(f"Processing transcription for: {participant}")
        print(f"File: {file_path}")
        
        try:
            # Check to see if transcription exists 
            json_output_path = os.path.join(output_dir, f'{participant}_transcription.json')

            my_file = Path(json_output_path)
            if my_file.is_file():
                print('Transcription JSON exists, skipping')
                continue

            
            # Load audio
            print('Loading audio')
            audio, _ = load_audio_from_file(file_path, sr)
            
            # Find all speech segments
            print('Finding all speech segments')
            segments = find_all_speech_segments(audio, sr, collar=collar)
            print(f"Found {len(segments)} segment(s)")
            
            # Process each segment
            all_transcriptions = []
            for i, (start_time, end_time, segment_audio) in enumerate(segments):
                print(f"Segment {i+1}/{len(segments)}: {start_time:.2f}s - {end_time:.2f}s ({end_time-start_time:.2f}s duration)")

                # Perform transcription for this segment
                print(f"Transcribing segment {i+1}")
                transcription = transcribe_audio(segment_audio, whisper_model, initial_prompt)
                
                # Adjust timestamps to be relative to the original video
                adjusted_transcription = adjust_transcription_timestamps(transcription, start_time)
                
                # Add segment metadata to transcription
                segment_data = {
                    'segment_number': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'transcription': adjusted_transcription
                }
                all_transcriptions.append(segment_data)
            
            # Save all transcriptions to JSON
            output_data = {
                'participant': participant,
                'total_segments': len(segments),
                'segments': all_transcriptions
            }
            
            with open(json_output_path, 'w', encoding='utf-8') as fp:
                json.dump(output_data, fp, ensure_ascii=False, indent=4)
            print(f"Saved transcription: {json_output_path}")
            print("---")


        except Exception as e:
            print(f"Error processing {participant}: {str(e)}")
            continue


def adjust_transcription_timestamps(transcription, offset):
    """
    Adjust all timestamps in a Whisper transcription by adding an offset.
    
    Args:
        transcription (dict): Whisper transcription result
        offset (float): Time offset in seconds to add to all timestamps
        
    Returns:
        dict: Transcription with adjusted timestamps
    """
    adjusted = transcription.copy()
    
    # Adjust segment-level timestamps
    if 'segments' in adjusted:
        adjusted['segments'] = []
        for segment in transcription['segments']:
            adjusted_segment = segment.copy()
            adjusted_segment['start'] = segment['start'] + offset
            adjusted_segment['end'] = segment['end'] + offset
            
            # Adjust word-level timestamps if present
            if 'words' in segment:
                adjusted_segment['words'] = []
                for word in segment['words']:
                    adjusted_word = word.copy()
                    adjusted_word['start'] = word['start'] + offset
                    adjusted_word['end'] = word['end'] + offset
                    adjusted_segment['words'].append(adjusted_word)
            
            adjusted['segments'].append(adjusted_segment)
    
    return adjusted

def main():
    parser = argparse.ArgumentParser(description='Transcribe video files using Whisper')
    parser.add_argument('--input_dirs', nargs='+', required=True,
                        help='Input directories containing video files')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for results')
    parser.add_argument('--model_size', default='medium',
                        help='Whisper model size (default: medium)')
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
        model_size=args.model_size,
        sr=args.sample_rate,
        device=args.device,
        audio_diary=args.audio_diary,
        onsite_interview=args.onsite_interview,
        offsite_interview=args.offsite_interview,
        cognition=args.cognition
    )


if __name__ == "__main__":
    main()