# Dataset

This app expects a `books.csv` in this directory. It isn't included in the repo.

**Dataset:** [Goodreads-books](https://www.kaggle.com/datasets/jealousleopard/goodreadsbooks) on Kaggle (user `jealousleopard`). It's a scrape of Goodreads book metadata, not an official Goodreads export, and Kaggle doesn't list a clear open license for redistribution - so rather than guess, I'm leaving it out and pointing you to the source instead.

**To get it:**
1. Download `books.csv` from the link above (free Kaggle account required).
2. Drop it in this folder as `data/books.csv`.

**Columns this app reads:** `title`, `authors`, `publisher`, `language_code`, `average_rating`. The Kaggle CSV also has `bookID`, `isbn`, `isbn13`, `num_pages`, `ratings_count`, `text_reviews_count`, and `publication_date`, which are loaded but not used.

Rows with parsing errors are skipped automatically (`on_bad_lines='skip'` in `data_loader.py`), which is how the original notebook handled the occasional malformed row in this CSV.

Any CSV with those five columns will work if you want to point the app at a different source.
