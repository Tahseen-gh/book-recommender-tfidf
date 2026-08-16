"""Entry point: python run.py [--backend tfidf|embeddings]"""

import argparse

from book_recommender.app import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--backend',
        choices=['tfidf', 'embeddings'],
        default='tfidf',
        help="Search backend to use. 'tfidf' (default) needs no model download; "
             "'embeddings' downloads a sentence-transformers model on first run.",
    )
    args = parser.parse_args()
    main(backend=args.backend)
