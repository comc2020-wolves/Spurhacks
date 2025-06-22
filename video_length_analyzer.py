import os
import sys
from pathlib import Path
from moviepy.editor import VideoFileClip
import time

def get_video_duration(video_path):
    """
    Get the duration of a video file in seconds.
    
    Args:
        video_path (str): Path to the video file
        
    Returns:
        float: Duration in seconds, or None if error
    """
    try:
        with VideoFileClip(video_path) as clip:
            return clip.duration
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
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

def analyze_videos_in_folder(folder_path):
    """
    Analyze all video files in a folder and display their lengths.
    
    Args:
        folder_path (str): Path to the folder containing videos
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
    print("-" * 80)
    
    total_duration = 0
    successful_files = 0
    
    # Process each video file
    for i, video_path in enumerate(video_files, 1):
        print(f"Processing {i}/{len(video_files)}: {video_path.name}")
        
        duration = get_video_duration(str(video_path))
        
        if duration is not None:
            formatted_duration = format_duration(duration)
            total_duration += duration
            successful_files += 1
            print(f"  Duration: {formatted_duration} ({duration:.2f} seconds)")
        else:
            print(f"  Duration: Error processing file")
        
        print()
    
    # Summary
    print("-" * 80)
    print(f"Summary:")
    print(f"  Total files processed: {len(video_files)}")
    print(f"  Successful: {successful_files}")
    print(f"  Failed: {len(video_files) - successful_files}")
    
    if successful_files > 0:
        print(f"  Total duration: {format_duration(total_duration)} ({total_duration:.2f} seconds)")
        print(f"  Average duration: {format_duration(total_duration / successful_files)} ({total_duration / successful_files:.2f} seconds)")

def main():
    """Main function to handle command line arguments and run the analyzer."""
    if len(sys.argv) != 2:
        print("Usage: python video_length_analyzer.py <folder_path>")
        print("Example: python video_length_analyzer.py C:\\Videos")
        return
    
    folder_path = sys.argv[1]
    print(f"Analyzing videos in: {folder_path}")
    print()
    
    start_time = time.time()
    analyze_videos_in_folder(folder_path)
    end_time = time.time()
    
    print(f"\nAnalysis completed in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main() 