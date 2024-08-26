from newsapi import NewsApiClient
from transformers import BartTokenizer, BartForSequenceClassification
from config import *

api_key = news_key
query = "Artificial Intelligence"


# retrieve news articles
def fetch_articles(api_key, query, language="en", page_size=5):
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language=language, page_size=page_size)
    return articles["articles"]


# load pre-trained BART model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
model = BartForSequenceClassification.from_pretrained("facebook/bart-large-cnn")


def summarize_article(article_text):
    inputs = tokenizer.encode(
        "what company is this article about: " + article_text,
        return_tensors="pt",
        max_length=1024,
        truncation=True,
    )
    summary_ids = model.generate(
        inputs,
        max_length=150,
        min_length=50,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary


def generate_personalized_summaries(api_key, query):
    # fetch and process articles
    articles = fetch_articles(api_key, query)

    # summarize each article
    summaries = []
    for article in articles:
        if article["content"]:
            summary = summarize_article(article["content"])
            summaries.append({"title": article["title"], "summary": summary})
    return summaries


# generate summaries for articles
summaries = generate_personalized_summaries(api_key, query)

for summary in summaries:
    print(f"Title: {summary['title']}")
    print(f"Summary: {summary['summary']}")
