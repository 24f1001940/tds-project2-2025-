TDS Solver

Overview

TDS Solver is an API-based application designed to automatically answer graded assignment questions from the "Tools in Data Science" course at IIT Madras. The API processes user-provided questions and optional file attachments to extract and return the correct answer.

Features

Accepts questions as input via API.

Supports CSV and ZIP file uploads.

Extracts answers dynamically from provided files.

Returns responses in JSON format.

Deployed on Vercel for public access.

API Usage

Endpoint:

POST https://your-app.vercel.app/api/

Request Format:

Headers: Content-Type: multipart/form-data

Form Data:

question (string, required) - The question from the graded assignment.

file (file, optional) - A CSV or ZIP file containing the answer.

Example Request (Using cURL):

curl -X POST "https://your-app.vercel.app/api/" \
  -H "Content-Type: multipart/form-data" \
  -F "question=What is the value in the 'answer' column of the CSV file?" \
  -F "file=@extract.csv"

Example Response:

{
  "answer": "12345678dd90"
}

Deployment

This project is deployed on Vercel to ensure public accessibility. Once deployed, the API runs continuously without requiring manual execution.

Installation & Local Setup

To run the API locally:

Clone the repository:

git clone https://github.com/your-username/tds-solver.git
cd tds-solver

Install dependencies:

pip install -r requirements.txt

Run the Flask app:

flask run

License

This project is licensed under the MIT License. See the LICENSE file for details.

