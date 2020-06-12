from gensim.models import word2vec


def train():
    '''
    用gensim进行训练 设定min_count = 6, size = 600 最终结果保存到word2vec.txt中
    :return: none
    '''
    with open("train_en_new.txt", 'r', encoding="UTF-8") as f:
        word2 = set()
        cnt = 0
        sentences = []
        for line in f:
            if len(line) > 1:
                words = line.strip().split()
                for word in words:
                    if word not in word2:
                        word2.add(word)
                        cnt = cnt + 1
                sentences.append(line.strip().split())
        print("START")
        model = word2vec.Word2Vec(sentences, size = 600, sg = 1, min_count = 6)
        print("FINISH")
        model.wv.save_word2vec_format("word2vec.txt", binary=False)
        print(cnt)

if __name__ == "__main__":
    train()