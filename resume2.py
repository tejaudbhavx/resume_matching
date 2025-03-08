import os
import io
import json
import datetime
import fitz  # PyMuPDF for PDF
import docx
import boto3
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import cohere
from google import genai
from typing import List

# Load environment variables
load_dotenv()

# API Keys and configuration from .env file
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
S3_BUCKET_NAME = os.getenv('S3_BUCKET_NAME')
MONGO_URI = os.getenv('MONGO_URI')

# Initialize external services
co = cohere.Client(api_key=COHERE_API_KEY)
gemini_client = genai.Client(api_key=GEMINI_API_KEY)
s3_client = boto3.client('s3', aws_access_key_id=AWS_ACCESS_KEY, aws_secret_access_key=AWS_SECRET_KEY)

# MongoDB connection using the company URI (test database)
client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client['test']  # Using the 'test' database

# Collections for storing resumes
students_collection = db['students']
jobseekers_collection = db['jobseekers']

COHERE_EMBEDDING_MODEL = 'embed-english-v3.0'

# FastAPI instance
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "FastAPI is running!"}

# ------------------ S3 Functions ------------------

def upload_to_s3(file_content: bytes, file_name: str) -> str:
    """
    Uploads a file to Amazon S3 and returns the S3 URL.
    """
    try:
        s3_client.put_object(Bucket=S3_BUCKET_NAME, Key=file_name, Body=file_content)
        return f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{file_name}"
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {e}")

def get_resume_from_s3(file_name: str) -> bytes:
    """
    Retrieves a file from Amazon S3.
    """
    try:
        response = s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=file_name)
        return response['Body'].read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve file from S3: {e}")

# ------------------ Cohere & Gemini Functions ------------------

def fetch_embeddings(texts: List[str], embedding_type: str = 'search_document') -> List[List[float]]:
    """
    Fetches text embeddings using Cohere API.
    """
    try:
        results = co.embed(
            texts=texts,
            model=COHERE_EMBEDDING_MODEL,
            input_type=embedding_type
        ).embeddings
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cohere embedding fetch failed: {e}")

def synthesize_answer(question: str, context: List[str]) -> str:
    """
    Uses Gemini AI to extract experience & skills from resume text.
    """
    context_str = '\n'.join(context)
    prompt = f"""
    Extract ONLY the total years of experience and list of skills from the following document.
    ---------------------
    {context_str}
    ---------------------
    Provide the answer in the format:
    Years of Experience: <number> \n
    Skills: <comma-separated list>
    """
    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt]
        )
        return response.text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with Gemini API: {e}")

# ------------------ API Endpoint ------------------

@app.post("/extract-experience-skills/")
async def extract_experience_skills(
    file: UploadFile = File(...),
    user_type: str = Form(...)  # New form field: "student" or "jobseeker"
):
    """
    Handles resume uploads, extracts experience & skills, and stores data in MongoDB.
    The resume is stored in the 'students' collection if user_type is 'student'
    and in 'jobseekers' collection if user_type is 'jobseeker'.
    """
    try:
        file_extension = file.filename.split('.')[-1].lower()
        file_content = await file.read()
        # Define S3 path
        s3_file_path = f"resume/{file.filename}"
        
        # Check if resume exists (for simplicity, we check in both collections)
        existing_doc = students_collection.find_one({"file_name": file.filename}) or \
                       jobseekers_collection.find_one({"file_name": file.filename})
        if existing_doc:
            file_content = get_resume_from_s3(s3_file_path)
            print("Using existing resume from S3")
        else:
            s3_url = upload_to_s3(file_content, s3_file_path)
            print("New resume uploaded to S3")
        
        # Extract text from the file based on its type
        texts = []
        if file_extension == 'pdf':
            doc = fitz.open(stream=file_content, filetype='pdf')
            texts = [page.get_text() for page in doc]
        elif file_extension == 'docx':
            doc = docx.Document(io.BytesIO(file_content))
            texts = [para.text for para in doc.paragraphs]
        elif file_extension == 'txt':
            texts = file_content.decode('utf-8').splitlines()
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type")
        
        print(f"Extracted {len(texts)} pages/paragraphs from the document.")
        answer = synthesize_answer("Extract total years of experience and skills.", texts)
        print(f"Gemini response: {answer}")
        
        experience_match = None
        skills_match = None
        if "Years of Experience:" in answer:
            experience_match = answer.split("Years of Experience:")[1].split("\n")[0].strip()
        if "Skills:" in answer:
            skills_match = answer.split("Skills:")[1].strip()
        
        print(f"Experience: {experience_match}")
        print(f"Skills: {skills_match}")
        
        text_combined = "\n".join(texts)
        embeddings = fetch_embeddings([text_combined])
        
        resume_data = {
            "file_name": file.filename,
            "s3_url": f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_file_path}",
            "text": text_combined,
            "technical_skills": skills_match if skills_match else "N/A",
            "years_of_experience": experience_match if experience_match else "N/A",
            "embeddings": json.dumps(embeddings),
            "uploaded_at": datetime.datetime.utcnow()
        }
        
        # Decide collection based on user_type input
        user_type = user_type.lower().strip()
        if user_type == "student":
            students_collection.insert_one(resume_data)
            print("Record inserted in 'students' collection")
        elif user_type == "jobseeker":
            jobseekers_collection.insert_one(resume_data)
            print("Record inserted in 'jobseekers' collection")
        else:
            raise HTTPException(status_code=400, detail="Invalid user type. Must be 'student' or 'jobseeker'.")
        
        return JSONResponse(content={"document_id": file.filename, "answer": answer})
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing the document: {e}")
