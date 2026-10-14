import pandas as pd
import numpy as np
from dotenv import load_dotenv
from langchain_classic import text_splitter

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter
from langchain_core.documents import Document

import gradio as gr
from traitlets.utils import descriptions

load_dotenv()

books = pd.read_csv("books_with_emotions.csv")
books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"]
)

raw_documents = TextLoader("tagged_description.txt",
                           encoding="utf-8").load()
documents = [
    Document(page_content=line)
    for line in raw_documents[0].page_content.splitlines()
    if line.strip()
]
db_books = Chroma.from_documents(documents,OpenAIEmbeddings())


def retrieve_semantic_recommendations(
        query: str,
        category: str = None,
        tone: str = None,
        initial_top_k: int = 50,
        final_top_k: int = 16,
) -> pd.DataFrame:

    recs = db_books.similarity_search(query,initial_top_k)
    books_list = [int(rec.page_content.split(':')[0].strip('"')) for rec in recs]
    book_recs = books[books["isbn13"].isin(books_list)].head(final_top_k)

    if category!= "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].head(final_top_k)
    else:
        book_recs = book_recs.head(final_top_k)

    if tone == "Happy":
        book_recs.sort_values(by="joy",ascending=False,inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False,inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False,inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness",ascending=False,inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)

    return book_recs

def recommend_books(
    query: str,
    category: str,
    tone: str
):
    recommendations = retrieve_semantic_recommendations(query, category, tone)
    results = []

    for _, row in recommendations.iterrows():
        description = row["description"] if isinstance(row["description"], str) else "No description available."

        authors_split = row["authors"].split(";")
        if len(authors_split) == 2:
            authors_str = f"{authors_split[0]} and {authors_split[1]}"
        elif len(authors_split) > 2:
            authors_str = f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"
        else:
            authors_str = row["authors"]

        caption = f"{row['title']} by {authors_str}\n\n{description}"
        results.append((row["large_thumbnail"], caption))

    return results


categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All"] + ["Happy","Surprising","Angry","Sad","Suspenseful"]

with gr.Blocks() as dashboard:
    gr.Markdown("# Semantic Book Recommender",elem_classes="centered")

    with gr.Row():
        gr.Column(scale=1)
        with gr.Column(scale=4):
            with gr.Row():
                user_query = gr.Textbox(label="Please enter a description of a book:",
                                        placeholder="e.g. A story that will make me happy")
                category_dropdown = gr.Dropdown(choices = categories, label="Please select a category:",value="All")
                tone_dropdown = gr.Dropdown(choices = tones, label="Please select a tone:", value="All")
        gr.Column(scale=1)

    with gr.Row():
        gr.Column(scale=2)
        with gr.Column(scale=1, min_width=200):
            submit_button = gr.Button("Find a recommendation", variant="primary")
        gr.Column(scale=2)

    gr.Markdown("# Recommendations",elem_classes="centered")
    output = gr.Gallery(label="Recommended books", columns=8, rows=2,
                        height=500, object_fit="cover")


    submit_button.click(recommend_books,
                        inputs = [user_query,category_dropdown,tone_dropdown],
                        outputs= [output])


with open("style.css", encoding="utf-8") as f:
    css = f.read()

if __name__ == "__main__":
    dashboard.launch(theme=gr.themes.Glass(), css=css)