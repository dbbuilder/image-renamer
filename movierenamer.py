"""
MovieRenamer.py

This script renames movie files (e.g., MP4, MOV) in a specified directory based on extracted content from the first, middle, and last frames.
It supports various video formats and allows for preprocessing such as handling the --DATE parameter to prepend the date to the file name
and the --COMPRESS parameter to compress the video files.

Setup Instructions:
1. Install required packages:
   pip install opencv-python Pillow transformers
   (Optional: Install ffmpeg for video compression)
2. Run the script from the command line:
   python MovieRenamer.py <directory_path> [--DATE] [--COMPRESS]

Supported Video Formats: .mp4, .mov
"""

import os
import sys
import cv2
import subprocess
from datetime import datetime
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Global constant for supported video file formats
SUPPORTED_VIDEO_FORMATS = [".mp4", ".mov"]

# Initialize the BLIP model and processor for frame captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def extract_frame(video_path, frame_position):
    """Extract a specific frame from a video."""
    try:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_number = int(total_frames * frame_position)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            log_message(f"Error reading frame at position {frame_position} from {video_path}")
            return None
        return frame
    except Exception as e:
        log_message(f"Error extracting frame at position {frame_position} from {video_path}: {e}")
        return None

def frame_to_image(frame):
    """Convert an OpenCV frame to a PIL Image."""
    try:
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return image
    except Exception as e:
        log_message(f"Error converting frame to image: {e}")
        return None

def generate_caption_from_frame(frame):
    """Generate a caption for a given video frame using the BLIP model."""
    try:
        image = frame_to_image(frame)
        if not image:
            return ""
        inputs = processor(image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
        return caption.strip()
    except Exception as e:
        log_message(f"Error generating caption from frame: {e}")
        return ""

def resolve_conflict(directory, base_name, fileext):
    """Resolve naming conflicts by appending a numeric suffix to the base name."""
    counter = 1
    while True:
        new_filename = f"{base_name}{counter:03d}{fileext}"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        counter += 1

def log_message(message, log_filename="MovieRenamer_Log.txt", log_directory="."):
    """Log a message with a timestamp to a specified log file in the target directory."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_path = os.path.join(log_directory, log_filename)
    try:
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"Error writing to log file: {e}")

def get_video_creation_date(video_path):
    """Extract the creation or modification date from a video file."""
    try:
        # Use the file's modification date as a fallback
        modification_time = os.path.getmtime(video_path)
        return datetime.fromtimestamp(modification_time).strftime('%Y-%m-%d')
    except Exception as e:
        log_message(f"Error extracting date from {video_path}: {e}")
        return None

def compress_video(video_path):
    """Compress the video file using ffmpeg."""
    try:
        compressed_path = os.path.splitext(video_path)[0] + "_compressed.mp4"
        command = [
            "ffmpeg", "-i", video_path, "-vcodec", "libx264", "-crf", "28", "-preset", "fast", compressed_path
        ]
        subprocess.run(command, check=True)
        log_message(f"Compressed {video_path} to {compressed_path}")
        return compressed_path
    except Exception as e:
        log_message(f"Error compressing {video_path}: {e}")
        return None

def process_videos(directory, prepend_date=False, compress=False):
    """Process videos in the specified directory: rename them based on content from frames."""
    files_exist = any(
        file.lower().endswith(tuple(SUPPORTED_VIDEO_FORMATS))
        for file in os.listdir(directory)
    )
    
    if not files_exist:
        log_message("No video files found with supported extensions.", log_directory=directory)
        print("No video files found with supported extensions.")
        return  # Exit the function if no matching files are found

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(tuple(SUPPORTED_VIDEO_FORMATS)):
                video_path = os.path.join(root, file)
                fileext = os.path.splitext(video_path)[1].lower()

                # Extract frames from the video
                first_frame = extract_frame(video_path, 0.0)
                middle_frame = extract_frame(video_path, 0.5)
                last_frame = extract_frame(video_path, 0.9)

                # Generate captions for each frame
                captions = []
                for frame in [first_frame, middle_frame, last_frame]:
                    if frame is not None:
                        captions.append(generate_caption_from_frame(frame))

                # Combine captions, truncate to 47 characters, and PascalCase
                combined_text = " ".join(captions).strip()
                pascal_case_name = "".join(
                    word.capitalize() for word in combined_text.split()
                )[:47]  # Truncate to 47 characters

                # Extract the date from metadata (if available) or fallback to file modification date
                date_taken = get_video_creation_date(video_path) if prepend_date else None

                # Prepend date to the filename if needed
                if prepend_date and date_taken:
                    base_name = f"{date_taken}_{os.path.splitext(file)[0]}"
                else:
                    base_name = os.path.splitext(file)[0]

                # Always add the number suffix (001, 002, etc.)
                if pascal_case_name:
                    base_name = pascal_case_name
                    if prepend_date and date_taken:
                        base_name = f"{date_taken}_{base_name}"

                    new_name = resolve_conflict(root, base_name, fileext)
                    new_path = os.path.join(root, new_name)
                    try:
                        os.rename(video_path, new_path)
                        log_message(f"Renamed {file} to {new_name}", log_directory=directory)
                    except Exception as e:
                        log_message(f"Error renaming {file} to {new_name}: {e}", log_directory=directory)

                # Compress the video if the --COMPRESS option is used
                if compress:
                    compressed_path = compress_video(new_path)
                    if compressed_path:
                        try:
                            os.rename(compressed_path, new_path)
                            log_message(f"Compressed video saved as {new_name}", log_directory=directory)
                        except Exception as e:
                            log_message(f"Error replacing original video with compressed version: {e}", log_directory=directory)

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
