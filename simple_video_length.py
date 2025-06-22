import os
import sys
import subprocess
from pathlib import Path
import json

def get_video_duration_ffprobe(video_path):
    """
    Get video duration using ffprobe (requires ffmpeg to be installed).
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        float: Duration in seconds, or None if error
    """
    try:
        cmd = [
            'ffprobe', 
            '-v', 'quiet',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(video_path)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    except (subprocess.CalledProcessError, KeyError, ValueError, FileNotFoundError) as e:
        print(f"ffprobe error for {video_path}: {e}")
        return None

def format_duration(seconds):
    """
    Format duration in seconds to a readable string.
    
    Args:
        seconds (float): Duration in seconds
        
    Returns:
        str: Formatted duration string
    """
    if seconds is None:
        return "Error"
    
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    else:
        return f"{minutes:02d}:{secs:02d}"

def get_file_size_mb(file_path):
    """Get file size in MB."""
    try:
        size_bytes = os.path.getsize(file_path)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0

def analyze_videos_in_folder(folder_path, use_ffprobe=True):
    """
    Analyze all video files in a folder and display their lengths.
    
    Args:
        folder_path (str): Path to the folder containing videos
        use_ffprobe (bool): Whether to use ffprobe for duration (requires ffmpeg)
    """
    # Common video file extensions
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"Error: Folder '{folder_path}' does not exist.")
        return
    
    if not folder.is_dir():
        print(f"Error: '{folder_path}' is not a directory.")
        return
    
    # Find all video files
    video_files = []
    for file_path in folder.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in video_extensions:
            video_files.append(file_path)
    
    if not video_files:
        print(f"No video files found in '{folder_path}'")
        print(f"Supported formats: {', '.join(video_extensions)}")
        return
    
    print(f"Found {len(video_files)} video file(s) in '{folder_path}':")
    print("-" * 100)
    
    total_duration = 0
    successful_files = 0
    total_size = 0
    
    # Process each video file
    for i, video_path in enumerate(video_files, 1):
        print(f"Processing {i}/{len(video_files)}: {video_path.name}")
        
        file_size = get_file_size_mb(video_path)
        total_size += file_size
        
        if use_ffprobe:
            duration = get_video_duration_ffprobe(video_path)
        else:
            duration = None  # Will show as "N/A" for duration
        
        if duration is not None:
            formatted_duration = format_duration(duration)
            total_duration += duration
            successful_files += 1
            print(f"  Duration: {formatted_duration} ({duration:.2f} seconds)")
        else:
            print(f"  Duration: N/A (install ffmpeg for duration info)")
        
        print(f"  Size: {file_size:.2f} MB")
        print()
    
    # Summary
    print("-" * 100)
    print(f"Summary:")
    print(f"  Total files processed: {len(video_files)}")
    print(f"  Total size: {total_size:.2f} MB")
    
    if use_ffprobe:
        print(f"  Successful duration reads: {successful_files}")
        print(f"  Failed duration reads: {len(video_files) - successful_files}")
        
        if successful_files > 0:
            print(f"  Total duration: {format_duration(total_duration)} ({total_duration:.2f} seconds)")
            print(f"  Average duration: {format_duration(total_duration / successful_files)} ({total_duration / successful_files:.2f} seconds)")
    else:
        print("  Duration information requires ffmpeg to be installed")

def main():
    """Main function to handle command line arguments and run the analyzer."""
    if len(sys.argv) < 2:
        print("Usage: python simple_video_length.py <folder_path> [--no-ffprobe]")
        print("Example: python simple_video_length.py C:\\Videos")
        print("Example: python simple_video_length.py C:\\Videos --no-ffprobe")
        return
    
    folder_path = sys.argv[1]
    use_ffprobe = "--no-ffprobe" not in sys.argv
    
    print(f"Analyzing videos in: {folder_path}")
    if not use_ffprobe:
        print("Note: Duration information disabled (use ffmpeg for duration)")
    print()
    
    analyze_videos_in_folder(folder_path, use_ffprobe)

if __name__ == "__main__":
    main() 