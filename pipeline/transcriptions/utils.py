# utils for transcription and diarization
import subprocess
import numpy as np
import glob
import os
import pathlib
import torch
from pyannote.core import Timeline, Segment, Annotation
from silero_vad import load_silero_vad, get_speech_timestamps
import whisper_timestamped as whisper
from pyannote.audio import Model
from pyannote.audio.pipelines import SpeakerDiarization

## GET FILE PATHS FROM THE DATABASE
def load_audio_from_file(file_path, sr=16000):
    """
    Extract audio from video file using ffmpeg and return as numpy array.
    
    Args:
        file_path (str): Path to video file
        sr (int): Sample rate for output audio
        
    Returns:
        tuple: (audio_array, sample_rate)
    """
    command = [
        'ffmpeg',
        '-i', file_path,
        '-f', 'f32le',          # 32-bit float output
        '-acodec', 'pcm_f32le',
        '-ar', str(sr),
        '-ac', '1',
        '-loglevel', 'error',
        '-'
    ]
    process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    audio_bytes = process.stdout

    # Convert bytes to numpy array
    audio_np = np.frombuffer(audio_bytes, np.float32)

    # Normalize audio
    if np.max(np.abs(audio_np)) > 0:
        audio_np = audio_np / np.max(np.abs(audio_np))

    return audio_np, sr


def save_audio_as_mp3(audio_array, output_path, sr=16000):
    """
    Save audio array as MP3 file using ffmpeg.
    
    Args:
        audio_array (np.array): Audio data
        output_path (str): Output MP3 file path
        sr (int): Sample rate
    """
    # Ensure output directory exists
    pathlib.Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    command = [
        'ffmpeg',
        '-f', 'f32le',
        '-ar', str(sr),
        '-ac', '1',
        '-i', '-',
        '-codec:a', 'mp3',
        '-b:a', '192k',
        '-loglevel', 'error',
        '-y',  # Overwrite output file
        output_path
    ]
    
    process = subprocess.run(command, input=audio_array.tobytes(), check=True)


def find_longest_speech_segment(audio, sr=16000, collar=180):
    """
    Use Silero VAD to find the longest continuous region of voice activity.
    
    Args:
        audio (np.array): Audio data
        sr (int): Sample rate
        collar (int): Collar duration in seconds for merging segments
        
    Returns:
        tuple: (start_time, end_time, extracted_audio)
    """
    vad_model = load_silero_vad()
    
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=sr,
        return_seconds=True,
    )
    
    timeline = Timeline()
    for stamp in speech_timestamps:
        timeline.add(Segment(stamp['start'], stamp['end']))
    
    # Find the longest segment after applying collar
    chosen_segment = None
    max_duration = -1.0
    for segment in timeline.support(collar=collar).segments_list_:
        if segment.duration > max_duration:
            max_duration = segment.duration
            chosen_segment = segment
    
    if chosen_segment is None:
        # If no speech found, return the entire audio
        return 0, len(audio) / sr, audio
    
    # Extract audio segment
    start_sample = int(chosen_segment.start * sr)
    end_sample = int(chosen_segment.end * sr)
    extracted_audio = audio[start_sample:end_sample]
    
    return chosen_segment.start, chosen_segment.end, extracted_audio

def find_all_speech_segments(audio, sr=16000, collar=180):
    """
    Use Silero VAD to find all continuous regions of voice activity.
    
    Args:
        audio (np.array): Audio data
        sr (int): Sample rate
        collar (int): Collar duration in seconds for merging segments
        
    Returns:
        list: List of tuples (start_time, end_time, extracted_audio) for each segment
    """
    vad_model = load_silero_vad()
    
    speech_timestamps = get_speech_timestamps(
        audio,
        vad_model,
        sampling_rate=sr,
        return_seconds=True,
    )
    
    timeline = Timeline()
    for stamp in speech_timestamps:
        timeline.add(Segment(stamp['start'], stamp['end']))
    
    # Collect all segments after applying collar
    all_segments = []
    for segment in timeline.support(collar=collar).segments_list_:
        start_sample = int(segment.start * sr)
        end_sample = int(segment.end * sr)
        extracted_audio = audio[start_sample:end_sample]
        
        all_segments.append((segment.start, segment.end, extracted_audio))
    
    # If no speech found, return the entire audio as a single segment
    if not all_segments:
        return [(0, len(audio) / sr, audio)]
    
    return all_segments

def get_onsite_files(parent_paths):
    """
    Get all onsite video files from specified parent paths.
    
    Args:
        parent_paths (list): List of directory paths to search
        
    Returns:
        list: Sorted list of video file paths
    """
    video_files = []
    
    for parent_path in parent_paths:
        mov_files = glob.glob(os.path.join(parent_path, '**/*.[mM][oO][vV]'), recursive=True)
        mkv_files = glob.glob(os.path.join(parent_path, '**/*.[mM][kK][vV]'), recursive=True)
        video_files.extend(mov_files + mkv_files)
        
    return sorted(video_files)

def get_offsite_files(parent_paths):
    """
    Get all offsite video files from specified parent paths.
    
    Args:
        parent_paths (list): List of directory paths to search
        
    Returns:
        list: Sorted list of video file paths
    """
    video_files = []
    
    for parent_path in parent_paths:
        mp4_files = glob.glob(os.path.join(parent_path, '**/*.[mM][pP][4]'), recursive=True)
        mp4_files = [file for file in mp4_files if 'gvo' in file] #Gather joint audio
        # video_files.extend(mp4_files) 
        m4a_files = glob.glob(os.path.join(parent_path, '**/*.[mM][4][aA]'), recursive=True) #gather individual audio
        video_files.extend(mp4_files+m4a_files)
        
    return sorted(video_files)

def get_audio_files(parent_paths):
    """
    Get all audio files from specified parent paths.
    
    Args:
        parent_paths (list): List of directory paths to search
        
    Returns:
        list: Sorted list of video file paths
    """
    audio_files = []
    
    for parent_path in parent_paths:
        mp3_files = glob.glob(os.path.join(parent_path, '**/*.[mM][pP][3]'), recursive=True)
        audio_files.extend(mp3_files)
        
    return sorted(audio_files)

def get_cognition_files(parent_paths):
    """
    Get all cognition files from specified parent paths.
    
    Args:
        parent_paths (list): List of directory paths to search
        
    Returns:
        list: Sorted list of video file paths
    """
    video_files = []
    
    for parent_path in parent_paths:
        m4a_files = glob.glob(os.path.join(parent_path, '**/*.[mM][4][aA]'), recursive=True)
        m4a_files = [file for file in m4a_files if 'Cognition' in file] #Only transcribe cognition m4a 
        video_files.extend(m4a_files)
        
    return sorted(video_files)

def get_participant_id(file_path):
    """
    Extract participant ID from file path.
    Assuming files are read like /mnt/PREDiCTOR/PHOENIX/PROTECTED/PREDiCTOR/[StudyID]/something/something_else/my_file.xxx
    
    Args:
        file_path (str): Path to video file
        
    Returns:
        str: Participant ID (filename without extension)
    """

    return f'{pathlib.Path(file_path).stem}'

def setup_diarization_pipeline(token, device='cuda'):
    """
    Set up the pyannote diarization pipeline with default settings.
    
    Args:
        token (str): Hugging Face token for pyannote access
        device (str): Device to run on ('cuda' or 'cpu')
        
    Returns:
        SpeakerDiarization: Configured pipeline
    """
    
    # Create pipeline with default settings
    pipeline = SpeakerDiarization.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token
    )
    
    pipeline.to(torch.device(device))
    
    return pipeline

def setup_whisper_model(model_size='large-v2', device='cuda'):
    """
    Set up the Whisper model for transcription.
    
    Args:
        model_size (str): Whisper model size
        device (str): Device to run on
        
    Returns:
        WhisperModel: Loaded Whisper model
    """
    
    
    device_type = device
    return whisper.load_model(model_size, device=device_type)