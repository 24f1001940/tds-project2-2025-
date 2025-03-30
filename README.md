# TDS Solver  

## Overview  
**TDS Solver** is an API-based application developed by **Mohd Saqib** to automatically answer graded assignment questions from the *Tools in Data Science* course at **IIT Madras**. The API processes user-provided questions and optional file attachments to extract and return the correct answer.  

## Features  
✔ Accepts questions as input via API.  
✔ Supports **CSV** and **ZIP** file uploads.  
✔ Dynamically extracts answers from provided files.  
✔ Returns responses in **JSON** format.  
✔ **Deployed on Vercel** for public access.  

## API Usage  

### Endpoint  
```plaintext
POST https://tds-project2-2025-new-git-main-mohd-saqibs-projects-351cb01b.vercel.app/api
```

### Request Format  

**Headers:**  
`Content-Type: multipart/form-data`  

**Form Data:**  
- `question` *(string, required)* – The question from the graded assignment.  
- `file` *(file, optional)* – A **CSV** or **ZIP** file containing the answer.  

### Example Request (Using cURL)  
```bash
curl -X POST "https://tds-project2-2025-new-git-main-mohd-saqibs-projects-351cb01b.vercel.app/api" \
  -H "Content-Type: multipart/form-data" \
  -F "question=How many Wednesdays are there in the given dataset?" \
  -F "file=@dataset.csv"
```

### Example Response  
```json
{
  "answer": "883"
}
```

## Deployment  
This project is deployed on **Vercel**, ensuring public accessibility. The API runs continuously without requiring manual execution.  

## Installation & Local Setup  

To run the API locally:  

1️⃣ **Clone the repository:**  
```bash
git clone https://github.com/your-username/tds-solver.git
cd tds-solver
```

2️⃣ **Install dependencies:**  
```bash
pip install -r requirements.txt
```

3️⃣ **Run the Flask app:**  
```bash
flask run
```

## License  
This project is licensed under the **MIT License**. See the `LICENSE` file for details.  
