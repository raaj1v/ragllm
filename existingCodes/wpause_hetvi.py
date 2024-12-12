from google.oauth2 import service_account
from googleapiclient.discovery import build
from apiclient.http import MediaFileUpload, MediaIoBaseDownload
from deep_translator import GoogleTranslator
from langdetect import detect
import io
import concurrent.futures
import os
import time
import json
import hashlib
import pdfplumber
import pandas as pd
from docx import Document
from tabulate import tabulate
from multiprocessing import Pool
import xlrd
from openpyxl import load_workbook
import re
import logging
import random
import string
import subprocess
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from pdf2image import convert_from_path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def format_to_markdown(text, source_type):
    """
    Convert extracted text to properly formatted markdown
    """
    # Split text into lines
    lines = text.split('\n')
    formatted_lines = []
    in_paragraph = False
    
    for line in lines:
        line = line.strip()
        
        # Skip empty lines
        if not line:
            if in_paragraph:
                formatted_lines.append('')
                in_paragraph = False
            continue
            
        # Detect and format headers
        if source_type == 'pdf':
            # Detect potential headers based on font size or styling (simplified)
            if len(line) < 100 and line.isupper():
                formatted_lines.append(f'## {line.title()}')
                formatted_lines.append('')
                continue
        
        # Format lists
        if line.startswith(('•', '-', '*', '○', '·')):
            formatted_lines.append(f'- {line[1:].strip()}')
            continue
        
        # Handle numbered lists
        if re.match(r'^\d+[\.\)]', line):
            formatted_lines.append(f'1. {line[line.find(" ")+1:]}')
            continue
            
        # Regular paragraphs
        if not in_paragraph:
            formatted_lines.append(line)
            in_paragraph = True
        else:
            # Append to existing paragraph
            formatted_lines[-1] = f'{formatted_lines[-1]} {line}'
    
    return '\n\n'.join(formatted_lines)

def format_table_to_markdown(table_data):
    """
    Convert table data to markdown format
    """
    if not table_data:
        return ""
        
    # Split the table data into lines
    table_lines = table_data.split('\n')
    formatted_tables = []
    current_table = []
    
    for line in table_lines:
        if line.strip():
            current_table.append(line)
        elif current_table:
            # Convert grid format to markdown
            if len(current_table) > 2:  # Ensure we have header and data
                df = pd.read_csv(io.StringIO('\n'.join(current_table)), sep='|')
                formatted_tables.append(tabulate(df, headers='keys', tablefmt='pipe'))
            current_table = []
    
    # Handle last table if exists
    if current_table:
        df = pd.read_csv(io.StringIO('\n'.join(current_table)), sep='|')
        formatted_tables.append(tabulate(df, headers='keys', tablefmt='pipe'))
    
    return '\n\n'.join(formatted_tables)

def extract_from_pdf(pdf_path):
    try:
        all_text_data = ""
        all_table_data = ""

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Extract text
                text_data = page.extract_text()
                if text_data:
                    all_text_data += f"{text_data}\n\n"

                # Extract tables
                tables = page.extract_tables()
                if tables:
                    for table in tables:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        all_table_data += tabulate(df, headers='keys', tablefmt='grid') + "\n\n"

        # Format text and tables to markdown
        formatted_text = format_to_markdown(remove_non_ascii(all_text_data), 'pdf')
        formatted_tables = format_table_to_markdown(remove_non_ascii(all_table_data))
        
        return formatted_text, formatted_tables
    except Exception as e:
        return f"Error extracting PDF: {e}", ""

def extract_from_html(html_path):
    try:
        with open(html_path, 'r', encoding='utf-8') as file:
            soup = BeautifulSoup(file, 'html.parser')
            
            # Process headings
            for tag in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
                level = int(tag.name[1])
                tag.replace_with(f"{'#' * level} {tag.text}\n\n")
            
            # Process lists
            for ul in soup.find_all('ul'):
                for li in ul.find_all('li'):
                    li.replace_with(f"- {li.text}\n")
                    
            for ol in soup.find_all('ol'):
                for i, li in enumerate(ol.find_all('li'), 1):
                    li.replace_with(f"{i}. {li.text}\n")
            
            # Process tables
            tables_data = []
            for table in soup.find_all('table'):
                df = pd.read_html(str(table))[0]
                tables_data.append(tabulate(df, headers='keys', tablefmt='pipe'))
                table.decompose()
            
            # Get remaining text
            text = soup.get_text(separator='\n')
            formatted_text = format_to_markdown(remove_non_ascii(text), 'html')
            formatted_tables = '\n\n'.join(tables_data)
            
            return formatted_text, formatted_tables
    except Exception as e:
        return f"Error extracting HTML: {e}", ""

def process_file(file_path, folder_data):
    filename = os.path.basename(file_path)
    file_hash = calculate_file_hash(file_path)

    file_data = {"text": "", "tables": "", "error": ""}

    try:
        if filename.lower().endswith(".pdf"):
            text_data, table_data = extract_from_pdf(file_path)
            file_data["text"] = text_data
            file_data["tables"] = table_data
        elif filename.lower().endswith(".docx") or filename.lower().endswith(".doc"):
            text_data, table_data = process_document(file_path)
            file_data["text"] = format_to_markdown(text_data, 'doc')
            file_data["tables"] = format_table_to_markdown(table_data)
        elif filename.lower().endswith((".jpg", ".jpeg", ".png", ".gif")):
            extracted_text = process_image(file_path)
            file_data["text"] = format_to_markdown(extracted_text, 'image')
        elif filename.lower().endswith(".html"):
            text_data, table_data = extract_from_html(file_path)
            file_data["text"] = text_data
            file_data["tables"] = table_data
        else:
            file_data["error"] = "Unsupported file format"
    except Exception as e:
        file_data["error"] = f"Error processing {filename}: {e}"

    if file_hash in folder_data:
        new_filename = f"{filename}_{''.join(random.choices(string.ascii_letters + string.digits, k=6))}"
        folder_data[new_filename] = file_data
    else:
        folder_data[filename] = file_data

def write_to_file(output_file, folder_data):
    """
    Write the processed data to file in markdown format
    """
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for filename, file_data in folder_data.items():
                # Write filename as header
                f.write(f"# {filename}\n\n")
                
                # Write text content if exists
                if file_data["text"]:
                    f.write(file_data["text"] + "\n\n")
                
                # Write table content if exists
                if file_data["tables"]:
                    f.write("## Tables\n\n")
                    f.write(file_data["tables"] + "\n\n")
                
                # Write errors if any
                if file_data["error"]:
                    f.write(f"**Error:** {file_data['error']}\n\n")
                
                # Add separator between files
                f.write("---\n\n")
    except Exception as e:
        logger.error(f"Error writing to file {output_file}: {e}")

def process_directory(directory_path, output_base_path, timings_file):
    folder_timings = {}
    merged_folders = {}

    for root, dirs, files in os.walk(directory_path):
        folder_start_time = time.time()
        folder_data = {}
        folder_count = 0

        match = re.match(r"(\d{8})", os.path.basename(root))
        if not match:
            continue

        folder_prefix = match.group(1)
        
        if folder_prefix not in merged_folders:
            final_dir_name = folder_prefix
            output_dir = os.path.join(output_base_path, final_dir_name)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f'{final_dir_name}.txt')
            merged_folders[folder_prefix] = output_file
        else:
            output_file = merged_folders[folder_prefix]

        for filename in files:
            file_path = os.path.join(root, filename)
            process_file(file_path, folder_data)
            folder_count += 1

        write_to_file(output_file, folder_data)

        folder_elapsed_time = time.time() - folder_start_time
        folder_timings[root] = folder_elapsed_time

        with open(timings_file, 'a') as f:
            f.write(f"{root}: Processed {folder_count} files in {folder_elapsed_time:.2f} seconds.\n")

    return folder_timings