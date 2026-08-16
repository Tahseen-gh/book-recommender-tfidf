# Book Recommendation Engine

Type in a book title and get back the 10 closest matches, ranked by how similar their text is to what you typed. It's a content-based recommender: no user accounts, no click history, no collaborative filtering. Just TF-IDF vectors over book metadata and cosine similarity between them, served through a one-box Gradio UI.

![Gradio interface showing a search for "Harry Potter" and its top 10 results](docs/gradio_interface.png)

*Screenshot is a search against a small hand-built sample of 50 well-known books, used here for demonstration. The real dataset (see below) has around 11,000.*

## What it does

Every book in the dataset gets its title, authors, publisher, and language code cleaned (strip punctuation, lowercase, drop stopwords) and turned into TF-IDF vectors. When you search, your query goes through the same cleaning step, gets vectorized, and gets compared against every book's title vector with cosine similarity. The 10 highest-scoring books come back as a table with title, author, and average rating.

The stopword filter re-fetches NLTK's word list on every word it checks rather than loading it once per query. Fine at this dataset's size, would be worth caching if this ever ran over something much bigger.

## Why combine title, author, and publisher into one field

Two books can be alike in ways a title alone won't show: same author, same publisher, sometimes the same edition of the same language. So instead of vectorizing titles in isolation, I clean title, authors, publisher, and language_code separately and then concatenate them into one `combined_text` field per book before fitting a second, wider TF-IDF vectorizer over it (capped at 5000 features, versus no cap on the title-only vectorizer). The goal was to let the model pick up on "these books travel together" signals that pure title overlap would miss, without hand-engineering author or publisher weighting.

Language code turned out to be a weak addition. Almost every row in the dataset is `eng`, so it barely moves the vector. I left it in because it was cheap to include and it does help on the rare non-English rows.

One honest gap: the combined vectorizer and its full book-to-book similarity matrix get computed at startup, but the live search path doesn't use them. `get_recommendations` takes a `cosine_sim` argument that defaults to that combined matrix and never reads it in the function body - scoring runs through the title-only vectorizer instead. So today, search is really title-similarity search, not the title+author+publisher search the design intended. Wiring the combined matrix into scoring, or dropping it if title-only search turns out to be enough on its own, is the next thing to fix.

## How TF-IDF and cosine similarity fit together here

Two `TfidfVectorizer` instances get fit at startup: one on `combined_text` (described above), one on cleaned titles only. Cosine similarity is what actually drives search - a cleaned query is vectorized with the title vectorizer, compared against every title vector in the corpus, and `argsort` picks the top 10 by score. There's no similarity floor, so even a query with zero real overlap with anything in the corpus still gets back 10 books - whatever ranks highest by default. The UI has a "no similar books found" message for an empty result, but because scoring always returns exactly 10 rows, that message can never actually fire.

## Dataset

Not included in this repo. It's the [Goodreads-books dataset](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) on Kaggle - a scrape of Goodreads metadata rather than an official export, and Kaggle doesn't list a clear license for redistribution, so I'm pointing to the source instead of shipping a copy. Download `books.csv` yourself and drop it in `data/` (see `data/README.md` for the exact columns this app reads). Any CSV with `title`, `authors`, `publisher`, `language_code`, and `average_rating` columns will work.

## Running it

```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# put books.csv in data/ - see data/README.md
python run.py
```

Opens at `http://127.0.0.1:7860`. First run downloads the NLTK stopwords corpus automatically; after that it's cached locally and startup doesn't touch the network. Tested on Python 3.11.

## Structure

```
book_recommender/
  data_loader.py   loading + text cleaning
  recommender.py   TF-IDF vectorizers + cosine similarity scoring
  app.py           Gradio interface
run.py             entry point - python run.py
data/              put books.csv here, not tracked in git
docs/              screenshot
```
