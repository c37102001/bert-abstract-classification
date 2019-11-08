from sklearn.feature_extraction.text import TfidfVectorizer


def identity_tokenizer(text):
    return text


def handle_tfidf(datas):
    vectorizer = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False)
    return vectorizer, vectorizer.fit_transform(datas)


def get_tfidf(datas):
    vectorizer, tfidf = handle_tfidf(datas)
    return data_fit_tfidf(datas, tfidf.toarray(), vectorizer.vocabulary_)


def data_fit_tfidf(datas, tfidf, vocab):
    tfidfs = []
    for idx, data in enumerate(datas):
        word_idxs = [vocab[word] for word in data]
        tfidfs.append([tfidf[idx, word_idx] for word_idx in word_idxs])
    return tfidfs
