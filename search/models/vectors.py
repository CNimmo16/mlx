import gensim.downloader

word_vectors = None

def get_vecs():
    global word_vectors

    if not word_vectors:
        print("Downloading word vectors...")
        word_vectors = gensim.downloader.load('word2vec-google-news-300')
        print("Done")
    return word_vectors
