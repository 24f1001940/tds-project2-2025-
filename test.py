import requests
import json
API_URL = "http://127.0.0.1:8000/api/"

questions = [
    "vs code version",
    "make http request with uv",
    "prettier sha256sum",
    "google sheets formula result",
    "excel formula result",
    "hidden input secret value",
    "import html to google sheets",
    "count wednesdays",
    "get an llm to say yes" ,
    "function calling",
    "vector databases",
    "embedding similarity",
    "llm embeddings",
    "convert it into single json object",
    "paste the json at tools-in-data-science.pages.dev/jsonhash",
    "find all <div>s having a foo class",
    "sum up all the values where the symbol matches Š or Œ or ™",
    "create a github account",
    "replace all IITM with IIT Madras",
    "list all files in the folder along with their date and file size",
    "move all files under folders into an empty folder",
    "how many lines are different between a.txt and b.txt",
    "total sales of all the items in the gold ticket type",
    "scrape imdb movies",
    "scrape the bbc weather api",
    "find the bounding box of a city",
    "search hacker news",
    "find newest github user",
    "create a scheduled github action",
    "extract tables from pdf",
    "wikipedia outline",
    "convert a pdf to markdown",
    "clean up excel sales data",
    "clean up student marks",
    "apache log requests",
    "apache log downloads",
    "clean up sales data",
    "parse partial json",
    "extract nested json keys",
    "duckdb social media interactions",
    "transcribe a youtube video",
    "reconstruct an image",
    "sort this json array of objects by the value of the age field",
    "write documentation in markdown",
    "compress image losslessly",
    "host your portfolio pages in github",
    "google colab authentication result",
    "image brightness pixel count",
    "deploy a python api in vercel",
    "create a github action",
    "push an image to docker hub",
    "write a fastapi server to serve data",
    "ngrok url",
    "llm sentiment analysis",
    "llm token cost",
    "generate addresses with LLMs",
    "llm vision"
]


for question in questions:
    try:
        response = requests.post(API_URL, data={"question": question})
        response.raise_for_status()  # Raise an error for bad responses (4xx, 5xx)

        try:
            json_response = response.json()
            print(f"Question: {question}\nAnswer: {json_response}\n")
        except json.JSONDecodeError:
            print(f"⚠️ ERROR: Invalid JSON response for question: {question}")
            print(f"Response Text: {response.text}\n")

    except requests.exceptions.RequestException as e:
        print(f"❌ HTTP ERROR: {e}")
