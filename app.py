from flask import Flask, render_template, request
import os
import pandas as pd
import pdfplumber
import json
import yaml
import pyarrow.parquet as pq
import PyPDF2
import fitz  # PyMuPDF
import spacy

# Load spaCy NER model
nlp = spacy.load('en_core_web_sm')

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def extract_pdf_with_layout(pdf_file):
    """Extract text from PDF and try to retain the layout."""
    extracted_text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=2.0, y_tolerance=2.0)
            if text:
                extracted_text += text + "\n\n"
    return extracted_text

def extract_pdf_with_pymupdf(pdf_file):
    """Extract text with layout preservation using PyMuPDF."""
    doc = fitz.open(pdf_file)
    extracted_text = ""

    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        extracted_text += page.get_text("text") + "\n\n"

    doc.close()
    return extracted_text

def extract_entities(text):
    """Extract named entities from text using spaCy."""
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
    locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
    organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    return persons, locations, organizations

def process_file(file, file_type):
    """Process the file based on its type and return a DataFrame."""
    if file_type == "csv":
        return pd.read_csv(file)
    elif file_type in ["xls", "xlsx"]:
        return pd.read_excel(file)
    elif file_type == "json":
        data = json.load(file)
        return pd.json_normalize(data)
    elif file_type == "parquet":
        return pd.read_parquet(file)
    elif file_type in ["yaml", "yml"]:
        data = yaml.safe_load(file)
        return pd.json_normalize(data)
    elif file_type == "pdf":
        # Extract text with layout from PDF
        full_text = extract_pdf_with_layout(file)
        persons, locations, organizations = extract_entities(full_text)
        return pd.DataFrame({"PDF Text": [full_text], "Persons": [persons], "Locations": [locations], "Organizations": [organizations]})
    elif file_type == "txt":
        text = file.read().decode('utf-8')
        return pd.DataFrame(text.splitlines(), columns=["Text"])
    elif file_type in ["bin", "dat"]:
        binary_data = file.read()
        return pd.DataFrame({"Binary Data": [binary_data]})
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Get file extension and process accordingly
    file_ext = file.filename.split('.')[-1].lower()

    # Process the file based on type
    try:
        data_df = process_file(file, file_ext)
    except ValueError as e:
        return str(e), 400

    # Save processed data into an Excel file
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{file.filename.split('.')[0]}_converted.xlsx")
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        # Write data into Excel
        data_df.to_excel(writer, sheet_name="Data", index=False)

    return f"File has been uploaded and saved as {output_path}."

if __name__ == '__main__':
    app.run(debug=True)
