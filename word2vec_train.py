from gensim.models import word2vec


def train():
    with open("trainnew.txt", 'r', encoding="UTF-8") as f:
        sentences = []
        for line in f:
            if len(line) > 1:
                sentences.append(line.strip().split())
        print("START")
        model = word2vec.Word2Vec(sentences, size = 600, sg = 1, min_count = 1)
        print("FINISH")
        model.wv.save_word2vec_format("word2vec.txt", binary=False)

if __name__ == "__main__":
    train()