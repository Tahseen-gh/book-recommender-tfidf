# Book Recommendation Engine

Type in a book title and get back 10 similar books. It's a content-based recommender: no user accounts, no click history, no collaborative filtering. Just TF-IDF vectors over book metadata and cosine similarity between them, served through a one-box Gradio UI.

![Gradio interface showing a search for "Harry Potter" and its top 10 results](docs/gradio_interface.png)

*Screenshot is a search against a small hand-built sample of 50 well-known books, used here for demonstration. The real dataset (see below) has around 11,000.*

## What it does

Every book in the dataset gets its title, authors, publisher, and language code cleaned (strip punctuation, lowercase, drop stopwords) and turned into TF-IDF vectors. Search works in two steps: your query gets cleaned and compared against every book's title with cosine similarity to find the single closest title match, then that book gets compared against every other book using a second, wider similarity score built from title+author+publisher+language text. The 10 highest-scoring books from that second comparison come back as a table with title, author, and average rating. So a search is really "find the book closest to what I typed, then show me what's like that book" rather than a direct text match against your query.

Loading the full ~11,000-row dataset takes about 12 seconds: roughly 6.5s to clean the text columns, another 6s to fit both vectorizers and build the similarity matrix described below. Cleaning used to take about 150s on its own, because `clean_text` was reloading NLTK's stopword list from scratch on every single word it checked instead of once. Hoisting that load out to run once at import time is most of the difference.

## Why combine title, author, and publisher into one field

Two books can be alike in ways a title alone won't show: same author, same publisher, sometimes the same edition of the same language. So instead of vectorizing titles in isolation, I clean title, authors, publisher, and language_code separately and then concatenate them into one `combined_text` field per book before fitting a second, wider TF-IDF vectorizer over it (capped at 5000 features, versus no cap on the title-only vectorizer). The goal was to let the model pick up on "these books travel together" signals that pure title overlap would miss, without hand-engineering author or publisher weighting.

Language code turned out to be a weak addition. Almost every row in the dataset is `eng`, so it barely moves the vector. I left it in because it was cheap to include and it does help on the rare non-English rows.

The combined vectorizer and its full book-to-book similarity matrix (`cosine_sim_combined`, one row and column per book) get computed at startup and are what actually rank the results - see below.

## How TF-IDF and cosine similarity fit together here

Two `TfidfVectorizer` instances get fit at startup: one on `combined_text` (title+author+publisher+language, described above), one on cleaned titles only. They do different jobs. A typed query is free text that usually isn't an exact title, so there's no way to look it up directly in a book-to-book matrix - it has to be vectorized and compared fresh. That's what the title vectorizer is for: `get_recommendations` cleans the query, compares it against every title with cosine similarity, and takes the single best match (`argmax`, not top 10). From there, it switches to `cosine_sim_combined` and pulls that matched book's row out of the matrix - its precomputed combined-text similarity to every other book - and returns the top 10 from that row.

In other words: title similarity finds the book you probably meant, and combined similarity finds what's like it. Search for "The Hobbit" and it'll surface the other Tolkien books published by the same house even though their titles share no words with "Hobbit" at all - that's the combined-field matching working as intended.

Two things to know about the edges of this. There's no similarity floor, so a query with no real title overlap with anything still anchors on whatever title scores highest by default (index 0 if everything's tied at zero) and returns 10 books from that row - not meaningful, but never empty. And if your query matches an existing book closely enough to anchor on itself, that book is maximally similar to itself, so it usually lands as its own top result. The UI has a "no similar books found" message for an empty result, but because scoring always returns exactly 10 rows, that message can never actually fire.

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
