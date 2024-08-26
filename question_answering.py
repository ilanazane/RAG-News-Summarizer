from transformers import AutoTokenizer, BartForQuestionAnswering
import torch
from newsapi import NewsApiClient
from config import *


tokenizer = AutoTokenizer.from_pretrained("valhalla/bart-large-finetuned-squadv1")
model = BartForQuestionAnswering.from_pretrained(
    "valhalla/bart-large-finetuned-squadv1"
)

api_key = news_key
query = "machine learning"


# retrieve news articles
def fetch_articles(api_key, query, language="en", page_size=5):
    newsapi = NewsApiClient(api_key=api_key)
    articles = newsapi.get_everything(q=query, language=language, page_size=page_size)
    return articles["articles"]


def clean_answer(question, answer):
    if answer.startswith(question):
        return answer[len(question) :].strip()
    return answer


articles = fetch_articles(api_key, query)
titles = []

for article in articles:
    if article["content"]:
        texts = article["content"]
        titles.append(article["title"])

print(titles)

for i in range(len(titles)):
    print(titles[i])

    question, text = "What companies are mentioned in this sentence?", titles[i]

    inputs = tokenizer(question, text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    answer_start_index = outputs.start_logits.argmax()
    answer_end_index = outputs.end_logits.argmax()

    predict_answer_tokens = inputs.input_ids[
        0, answer_start_index : answer_end_index + 1
    ]
    answer = tokenizer.decode(predict_answer_tokens, skip_special_tokens=True)

    print(clean_answer(question, answer))
