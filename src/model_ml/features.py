from sklearn.feature_extraction.text import TfidfVectorizer


def build_vectorizer() -> TfidfVectorizer:
    # char_wb n-grams (2-4) work well for Vietnamese (tonal, no word-boundary spaces)
    # sublinear_tf dampens high-frequency terms
    return TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 4),
        max_features=50_000,
        sublinear_tf=True,
    )
