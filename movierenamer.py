"""
MovieRenamer.py

This script renames video files (MOV and MP4) in a specified directory based on captions generated 
from keyframes (first, middle, and last frames), converts MOV files to MP4, compresses the MP4 files, 
and prepends the date to the filenames based on metadata. The --COMPRESS argument converts all video 
files to compressed MP4 files for better storage efficiency. The --DATE argument prepends the date 
taken to the filename. Logging is performed in the target directory.

Setup Instructions:
1. Install required packages:
   pip install moviepy transformers opencv-python requests
2. Run the script from the command line:
   python MovieRenamer.py <directory_path> [--DATE] [--COMPRESS]

Supported Video Formats: .mp4, .mov
"""

import os
import sys
import re
import shutil
from datetime import datetime
from moviepy.editor import VideoFileClip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Global Constants
SUPPORTED_EXTENSIONS = [".mp4", ".mov"]
LOG_FILENAME = "MovieRenamer_Log.txt"
JPEG_QUALITY = 70
MAX_RESOLUTION = (1920, 1080)

# Initialize the BLIP model and processor for image captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def log_message(message, log_directory):
    """Log a message with a timestamp to a specified log file in the target directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(log_directory, LOG_FILENAME)
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def create_originals_folder(directory):
    """Create an 'ORIGINALS' folder in the specified directory."""
    originals_path = os.path.join(directory, "ORIGINALS")
    try:
        if not os.path.exists(originals_path):
            os.makedirs(originals_path)
        return originals_path
    except Exception as e:
        log_message(f"Error creating ORIGINALS folder in {directory}: {e}", log_directory=directory)
        return None

def move_to_originals(file_path, originals_folder, log_directory):
    """Move the original file to the 'ORIGINALS' folder, ensuring no filename conflicts."""
    try:
        # Ensure the destination filename is unique by appending a date-time stamp if needed
        base_name = os.path.splitext(os.path.basename(file_path))[0]
        extension = os.path.splitext(file_path)[1]
        dest_path = os.path.join(originals_folder, os.path.basename(file_path))

        # If the file already exists in the ORIGINALS folder, append a timestamp
        if os.path.exists(dest_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_base_name = f"{base_name}_{timestamp}"
            dest_path = os.path.join(originals_folder, new_base_name + extension)

        shutil.move(file_path, dest_path)
        log_message(f"Moved {file_path} to {dest_path}", log_directory=log_directory)
    except Exception as e:
        log_message(f"Error moving {file_path} to {originals_folder}: {e}", log_directory=log_directory)

def convert_mov_to_mp4(video_path, originals_folder, log_directory):
    """Convert a MOV video to MP4 format and move the original MOV file to the 'ORIGINALS' folder."""
    try:
        mp4_path = os.path.splitext(video_path)[0] + ".mp4"
        clip = VideoFileClip(video_path)
        clip.write_videofile(mp4_path, codec="libx264", audio_codec="aac")
        move_to_originals(video_path, originals_folder, log_directory)
        log_message(f"Converted {video_path} to {mp4_path} and moved original MOV to {originals_folder}", log_directory=log_directory)
        return mp4_path
    except Exception as e:
        log_message(f"Error converting MOV to MP4 for {video_path}: {e}", log_directory=log_directory)
        return None

def compress_mp4(video_path, originals_folder, log_directory, quality=JPEG_QUALITY):
    """Compress an existing MP4 video file."""
    try:
        compressed_path = os.path.splitext(video_path)[0] + "_compressed.mp4"
        clip = VideoFileClip(video_path)
        clip.write_videofile(compressed_path, codec="libx264", audio_codec="aac", bitrate="500k")
        log_message(f"Compressed {video_path} to {compressed_path}", log_directory=log_directory)
        return compressed_path
    except Exception as e:
        log_message(f"Error compressing {video_path}: {e}", log_directory=log_directory)
        return None

def extract_keyframes(video_path):
    """Extract the first, middle, and last frames from a video file."""
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration

        # Extract keyframes (first, middle, last)
        first_frame = clip.get_frame(0)
        middle_frame = clip.get_frame(duration / 2)
        last_frame = clip.get_frame(duration - 1)

        return [first_frame, middle_frame, last_frame]
    except Exception as e:
        log_message(f"Error extracting keyframes from {video_path}: {e}", log_directory=os.path.dirname(video_path))
        return []

def generate_caption_for_frame(frame, log_directory):
    """Generate a caption for a single video frame using the BLIP model."""
    try:
        image = Image.fromarray(frame)
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        log_message(f"Error generating caption for frame: {e}", log_directory=log_directory)
        return ""

def generate_combined_caption(video_path, log_directory):
    """Generate a combined caption from the first, middle, and last frames of the video."""
    keyframes = extract_keyframes(video_path)
    captions = [generate_caption_for_frame(frame, log_directory) for frame in keyframes if frame is not None]
    return " ".join(captions)

def resolve_conflict(directory, base_name, fileext):
    """Resolve naming conflicts by appending a numeric suffix to the base name."""
    counter = 1
    while True:
        new_filename = f"{base_name}{counter:03d}{fileext}"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        counter += 1

def convert_mov_files(directory, originals_folder, log_directory):
    """Convert all MOV files in the directory to MP4 while preserving the original MOV files."""
    for root, _, files in os.walk(directory):
        for file in files:
            fileext = os.path.splitext(file)[1].lower()
            if fileext == ".mov":
                video_path = os.path.join(root, file)
                convert_mov_to_mp4(video_path, originals_folder, log_directory)

def generate_captions(directory, originals_folder, log_directory):
    """Generate captions for all video files in the directory."""
    for root, _, files in os.walk(directory):
        for file in files:
            fileext = os.path.splitext(file)[1].lower()
            if fileext in SUPPORTED_EXTENSIONS:
                video_path = os.path.join(root, file)
                # Skip files in the ORIGINALS folder
                if "ORIGINALS" in root or "DUPLICATES" in root:
                    continue
                caption = generate_combined_caption(video_path, log_directory)
                if caption:
                    pascal_case_name = "".join(
                        word.capitalize() for word in caption.split()
                    )[:47]
                    new_name = resolve_conflict(root, pascal_case_name, fileext)
                    try:
                        os.rename(video_path, os.path.join(root, new_name))
                        move_to_originals(video_path, originals_folder, log_directory)
                    except Exception as e:
                        log_message(f"Error renaming {file} to {new_name}: {e}", log_directory=log_directory)

def prepend_dates_to_filenames(directory, originals_folder, log_directory):
    """Prepend the date taken to the filenames based on the file's modification date."""
    for root, _, files in os.walk(directory):
        for file in files:
            fileext = os.path.splitext(file)[1].lower()
            if fileext in SUPPORTED_EXTENSIONS:
                video_path = os.path.join(root, file)
                # Skip files in the ORIGINALS and DUPLICATES folders
                if "ORIGINALS" in root or "DUPLICATES" in root:
                    continue
                # Use the file's modification date as a proxy for the video date
                date_taken = datetime.fromtimestamp(os.path.getmtime(video_path)).strftime('%Y-%m-%d')
                if date_taken and not re.match(r'^\d{4}-\d{2}-\d{2}_', file):
                    new_name = f"{date_taken}_{file}"
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(video_path, new_path)
                        move_to_originals(video_path, originals_folder, log_directory)
                    except Exception as e:
                        log_message(f"Error renaming {file} to {new_name}: {e}", log_directory=log_directory)

def compress_videos(directory, originals_folder, log_directory):
    """Compress all MP4 files in the directory while preserving quality."""
    for root, _, files in os.walk(directory):
        for file in files:
            fileext = os.path.splitext(file)[1].lower()
            if fileext == ".mp4" and "_compressed" not in file.lower():
                video_path = os.path.join(root, file)
                # Skip files in the ORIGINALS and DUPLICATES folders
                if "ORIGINALS" in root or "DUPLICATES" in root:
                    continue
                compressed_video_path = compress_mp4(video_path, originals_folder, log_directory)
                if compressed_video_path:
                    move_to_originals(video_path, originals_folder, log_directory)

def process_videos(directory, prepend_date=False, compress=False):
    """Process video files in the specified directory by running them through multiple passes."""
    
    originals_folder = create_originals_folder(directory)
    log_directory = directory
    
    # Pass 1: Convert MOV to MP4
    convert_mov_files(directory, originals_folder, log_directory)

    # Pass 2: Generate captions
    generate_captions(directory, originals_folder, log_directory)

    # Pass 3: Prepend dates to filenames
    if prepend_date:
        prepend_dates_to_filenames(directory, originals_folder, log_directory)

    # Pass 4: Compress MP4 files
    if compress:
        compress_videos(directory, originals_folder, log_directory)

if __name__ == "__main__":
    # Check for command-line arguments
    prepend_date = "--DATE" in sys.argv
    compress = "--COMPRESS" in sys.argv

    # Get the directory from the command-line argument if provided
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Prompt the user for the directory if not provided
        directory = input("Enter the directory path containing videos: ")

    if not os.path.isdir(directory):
        print(f"Error: The provided directory '{directory}' does not exist.")
        sys.exit(1)

    process_videos(directory, prepend_date, compress)
