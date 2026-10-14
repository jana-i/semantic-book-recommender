# Semantic Book Recommender

Find books by describing them in your own words. 
For example, type "a mystery set in a small town with a dark secret" and the app suggests matching books.

You can also filter results by fiction or non-fiction, and sort them by mood (happy, sad, suspenseful, etc.).

## What it does

- Searches books by meaning, not just keywords
- Labels books as fiction or non-fiction
- Detects the emotional tone of each book
- Shows everything in a simple web app

This project was initially created in Python 3.11. 
To run the project, the following dependencies are required:

*kagglehub
*pandas
*matplotlib
*seaborn
*python-dotenv
*langchain-community
*langchain-opencv
*langchain-chroma
*transformers
*gradio
*notebook
*ipywidgets

## Setup
To create the vector database, add a `.env` file in the root directory containing your OpenAI API key (`OPENAI_API_KEY=your_key_here`). 
The data used in this project comes from the [7k Books with Metadata](https://www.kaggle.com/datasets/dylanjcastillo/7k-books-with-metadata) dataset on Kaggle.
