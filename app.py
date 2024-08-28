from flask import Flask, request, redirect, url_for, render_template,jsonify
from pypdf import PdfReader
import numpy as np
import pandas as pd
import os
from dotenv import load_dotenv
import google.generativeai as genai
import json
import textwrap
import requests
from bs4 import BeautifulSoup
from docx import Document
import pytesseract
from PIL import Image
import pdfplumber
from pdf2image import convert_from_path
# from appibm import llama31

app = Flask(__name__)

load_dotenv()

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'xlsx', 'xls', 'docx','csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


model = 'models/embedding-001'
generation_config = {
  "temperature": 1,
  "top_p": 0.95,
  "top_k": 64,
  "max_output_tokens": 8192,
  "response_mime_type": "text/plain",
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
]

prompt_model = genai.GenerativeModel(
  model_name="gemini-1.5-flash",
  generation_config=generation_config,
  safety_settings = safety_settings,
  # See https://ai.google.dev/gemini-api/docs/safety-settings
  system_instruction="Your name is Angel. Your role is to find the best and most relevant answer with step by step to the user's question.",
)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def upload_file():
    files = os.listdir(app.config['UPLOAD_FOLDER'])
    return render_template('index.html',files=files)

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = textwrap.dedent("""DOCUMENT: {relevant_passage}
QUESTION: {query}
INSTRUCTIONS: Answer the user's question using the text from the DOCUMENT above. Keep your answer grounded in the facts from the DOCUMENT. If the DOCUMENT does not contain the information needed to answer the question, find external sources and return the answer from those external sources, noting that the answer comes from an external source and not the PDF.
  """).format(query=query, relevant_passage=escaped)

  return prompt

def find_best_passage(query, dataframe, top_n=5):
  query_embedding = genai.embed_content(model=model,
                                        content=query,
                                        task_type="retrieval_query")
  dot_products = np.dot(np.stack(dataframe['Embeddings']), query_embedding["embedding"])
  # idx = np.argmax(dot_products)
  # return dataframe.iloc[idx]['Data']
  top_indices = np.argsort(dot_products)[::-1][:top_n]
  top_passages = dataframe.iloc[top_indices]['Data'].tolist()
  result = ''.join(top_passages)
  return result

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_message = request.json.get('message')
        data_str = request.json.get('data')
        data = json.loads(data_str)
        #print("Type",type(data))
        model = 'models/embedding-001'
        embeddings_list = genai.embed_content(model=model,content=user_message,task_type="retrieval_query")
        df = pd.DataFrame(data)
        passage = find_best_passage(user_message, df)
        #print(passage)
        # prompt= f"{passage} \n Question - {user_message}."
        prompt = make_prompt(user_message, passage)
        response = prompt_model.generate_content(prompt)

        #llama = llama31(user_message, passage)

        return jsonify({'response': response.text,'reference':passage})
    except json.JSONDecodeError as e:
        return jsonify({'response': 'Invalid JSON format', 'reference': str(e)}), 400
    except KeyError as e:
        return jsonify({'response': 'Missing key in the JSON request', 'reference': str(e)}), 400
    except Exception as e:
        return jsonify({'response': 'An error occurred', 'reference': str(e)}), 500

@app.route('/chat_2', methods=['POST'])
def chat_2():
    user_message = request.json.get('message')
    # prompt= f"{passage} \n Question - {user_message}."
    prompt = "Please find the best answer to my question.\nQUESTION -"+user_message
    response = prompt_model.generate_content(prompt)

    return jsonify({'response': response.text})



def embed_fn(title):
  return genai.embed_content(model=model,
                             content=title,
                             task_type="retrieval_document"
                             )["embedding"]

@app.route('/upload', methods=['GET', 'POST'])
def uploader():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also submits an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            ext = filename.rsplit('.', 1)[1].lower()
            if ext == 'pdf':
                content = read_pdf(filepath)
            elif ext == 'csv':
                content = read_csv(filepath)
            elif ext in ['xlsx', 'xls']:
                content = read_excel(filepath)
            elif ext == 'docx':
                content = read_docs(filepath)
            elif ext == 'txt':
                content = read_txt(filepath)
            else:
                return redirect(request.url)
            
            chunk_size=500
            content_chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
            df = pd.DataFrame(content_chunks)
            df.columns = ['Data']
            df['Embeddings'] = df.apply(lambda row: embed_fn(row['Data']), axis=1)
            embeddings_list = df[['Data', 'Embeddings']].to_dict(orient='records')
            
            return jsonify(embeddings_list)
            #return jsonify({'Data':content})
    return redirect(url_for('upload_file'))

####################################
def is_pdf_text_based(pdf_path):
    """Check if a PDF contains readable text."""
    try:
        # Attempt to read text using pdfplumber
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text and text.strip():
                    return True
    except Exception as e:
        print(f"Error checking PDF text content: {e}")   
    return False

def read_pdf(file):
    """Extract text from a PDF file, whether it's text-based or image-based."""
    # if is_pdf_text_based(file):
    #     reader = PdfReader(file)
    #     pdf_content = ''
    #     for page in reader.pages:
    #         pdf_content += page.extract_text()    
    #     return pdf_content
    # else:
    # output_dir = "output_text"
    # os.makedirs(output_dir, exist_ok=True)
    images = convert_from_path(file)
    text = ' '
    for image in images:
            # image_path = os.path.join(output_dir, f'page_1.png')
            # image.save(image_path, 'PNG')
            # sample_file = genai.upload_file(path=image_path,display_name="Jetpack drawing")
            # print(f"Uploaded file '{sample_file.display_name}' as: {sample_file.uri}")
            # break
        text += pytesseract.image_to_string(image)
    return text
###########################
def read_csv(file):
    df = pd.read_csv(file)
    rows_as_strings = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
    combined_string = '\n'.join(rows_as_strings)
    return combined_string

def read_txt(file):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

    return content
def read_excel(file):
    df = pd.read_excel(file)
    rows_as_strings = df.astype(str).apply(lambda row: ' '.join(row), axis=1).tolist()
    combined_string = '\n'.join(rows_as_strings)
    return combined_string

def read_docs(file):
    doc = Document(file)
    content=''
    for para in doc.paragraphs:
        content += para.text
    return content 



##### URL CHECK 
@app.route('/check_url', methods=['POST'])
def check_url():
    data = request.get_json()
    url = data.get('url')
    
    try:
        response = requests.get(url)
        if response.status_code == 200:
            print('valid')
            return jsonify({'valid': True, 'message': 'ok'})
        else:
            return jsonify({'valid': False, 'message': 'URL is not accessible (status code: {})'.format(response.status_code)})
    except requests.exceptions.RequestException as e:
        return jsonify({'valid': False, 'message': 'URL is not valid or accessible: {}'.format(e)})

## SCRAP CONTENT
@app.route('/scrapehtml', methods=['POST'])
def scrapehtml():
    data = request.json
    url = data.get('url')
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.get_text(separator='\n', strip=True)
    except requests.exceptions.RequestException as e:
        return jsonify({'error': str(e)})
    
    chunk_size=500
    urlchunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    df = pd.DataFrame(urlchunks)
    df.columns = ['Data']
    df['Embeddings'] = df.apply(lambda row: embed_fn(row['Data']), axis=1)
    embeddings_list = df[['Data', 'Embeddings']].to_dict(orient='records') 
    return jsonify(embeddings_list)

###########
@app.route('/delete/<filename>', methods=['DELETE'])
def delete_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if os.path.exists(file_path):
        os.remove(file_path)
        return '', 204
    else:
        return '', 404


if __name__ == '__main__':
    app.run(debug=True)
