import os
import sys
import re
import unicodedata
import uuid
from datetime import datetime
from PyPDF2 import PdfReader
from transformers import pipeline, AutoTokenizer

# Initialize the summarization model and tokenizer with a specified model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def contains_guid(filename):
    # Regular expression pattern to match a GUID/UUID
    pattern = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', re.IGNORECASE)
    match = pattern.search(filename)
    return match is not None

def extract_text_and_metadata_from_pdf(file_path):
    text = ""
    title = ""
    description = ""
    try:
        reader = PdfReader(file_path)
        
        # Extract text content from each page
        for page in reader.pages:
            text += page.extract_text()

        # Extract metadata if available
        metadata = reader.metadata
        
        title = metadata.get('/Title', '') if metadata else ''
        description = metadata.get('/Subject', '') if metadata else ''  # /Subject is often used as a description field

        return text.strip(), title.strip(), description.strip()

    except Exception as e:
        log_message(f"Error extracting text and metadata from {file_path}: {e}", os.path.dirname(file_path))
        return "", "", ""

def remove_special_characters(text):
    # Remove all characters except letters, numbers, and spaces
    cleaned_text = re.sub(r'[^A-Za-z0-9\s]', '', text)
    return cleaned_text

def ignore_dates(text):
    # Remove "Date:" or "date:" (case-insensitive) and date-like patterns (e.g., 2023-08-14, 14/08/2023)
    cleaned_text = re.sub(r'\bDate:\b', '', text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b', '', cleaned_text)
    cleaned_text = re.sub(r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b', '', cleaned_text)
    return cleaned_text

def resolve_conflict(directory, base_name, fileext):
    counter = 1
    while True:
        new_filename = f"{base_name}{counter:03d}{fileext}"
        if not os.path.exists(os.path.join(directory, new_filename)):
            return new_filename
        counter += 1

def normalize_filename(filename):
    # Normalize the filename by converting non-standard characters to ASCII and removing others
    normalized_filename = unicodedata.normalize('NFKD', filename).encode('ascii', 'ignore').decode('ascii')
    normalized_filename = re.sub(r'[^A-Za-z0-9\.\-_]', '', normalized_filename)  # Keep alphanumeric, dot, hyphen, and underscore
    return normalized_filename

def log_message(message, log_directory):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_filename = "PDFRenamer_Log.txt"
    log_path = os.path.join(log_directory, log_filename)
    
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

def truncate_text_to_max_tokens(text, max_length):
    # Tokenize the text
    tokens = tokenizer(text, truncation=True, max_length=max_length, return_tensors='pt')
    
    # Decode back to string
    truncated_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    
    return truncated_text

def summarize_content(text, log_dir):
    try:
        # Truncate text to fit within the model's maximum token length
        truncated_text = truncate_text_to_max_tokens(text, 1024)
        
        # Summarize the truncated text
        summary = summarizer(truncated_text, max_length=60, min_length=25, do_sample=False)
        return summary[0]['summary_text'].strip()
    except Exception as e:
        log_message(f"Error summarizing content: {e}", log_dir)
        return ""

def rename_untitled_file(root, file):
    base_name = "Document"
    fileext = ".pdf"
    new_name = resolve_conflict(root, base_name, fileext)
    os.rename(os.path.join(root, file), os.path.join(root, new_name))
    log_message(f"Renamed untitled file {file} to {new_name}", root)
    return new_name

def process_pdfs(directory):
    # Check if any supported PDF files exist in the directory
    files_exist = any(
        file.lower().endswith(".pdf") for file in os.listdir(directory)
    )
    
    if not files_exist:
        log_message("No PDF files found with supported extensions.", directory)
        print("No PDF files found with supported extensions.")
        return  # Exit the function if no matching files are found

    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".pdf"):
                # Normalize the filename and rename the file if necessary
                normalized_file_name = normalize_filename(file)
                original_file_path = os.path.join(root, file)
                normalized_file_path = os.path.join(root, normalized_file_name)

                if original_file_path != normalized_file_path:
                    os.rename(original_file_path, normalized_file_path)
                    log_message(f"Renamed {file} to {normalized_file_name}", root)

                # Handle "Untitled" files
                if normalized_file_name.lower().startswith("untitled"):
                    normalized_file_name = rename_untitled_file(root, normalized_file_name)

                # Continue with the normalized file name
                file_name = os.path.splitext(normalized_file_name)[0]
                if contains_guid(file_name):
                    pdf_path = os.path.join(root, normalized_file_name)
                    fileext = ".pdf"

                    # Extract text and metadata from PDF
                    pdf_text, pdf_title, pdf_description = extract_text_and_metadata_from_pdf(pdf_path)

                    # Ignore dates in text, title, and description
                    pdf_text = ignore_dates(pdf_text)
                    pdf_title = ignore_dates(pdf_title)
                    pdf_description = ignore_dates(pdf_description)

                    # Remove special characters
                    cleaned_text = remove_special_characters(pdf_text)
                    cleaned_title = remove_special_characters(pdf_title)
                    cleaned_description = remove_special_characters(pdf_description)

                    # Combine cleaned text, title, and description
                    combined_text = f"{cleaned_title} {cleaned_description} {cleaned_text}".strip()

                    # Log the entire combined text before summarization
                    log_message(f"Combined text before summarization for {file}: {combined_text}", root)

                    # Summarize the content
                    summary = summarize_content(combined_text, root)
                    log_message(f"Summary for {file}: {summary}", root)

                    # Convert the summary to PascalCase and truncate to 47 characters
                    pascal_case_name = "".join(
                        word.capitalize() for word in summary.split()
                    )[:47]  # Truncate to 47 characters

                    if pascal_case_name:
                        base_name = pascal_case_name
                        new_name = resolve_conflict(root, base_name, fileext)
                        new_path = os.path.join(root, new_name)
                        os.rename(pdf_path, new_path)
                        log_message(f"Renamed {file} to {new_name}", root)

if __name__ == "__main__":
    # Default to the current user's Downloads directory
    default_directory = os.path.join(os.path.expanduser("~"), "Downloads")

    # Get the directory from the command-line argument if provided
    if len(sys.argv) > 1:
        directory = sys.argv[1]
    else:
        # Use the default Downloads directory if not provided
        directory = default_directory

    if not os.path.isdir(directory):
        print(f"Error: The provided directory '{directory}' does not exist.")
        sys.exit(1)

    process_pdfs(directory)
