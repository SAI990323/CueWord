import numpy as np

sentence_len = 44
vec_len = 600


# 按照论文所说 一句话最多44个词 不足补0 超过截断

def read_word2vec(file):
    with open(file, 'r', encoding='utf-8') as f:
        word2vec = {}
        cnt = 0
        for line in f:
            if cnt == 0:
                cnt = 1
                continue
            line = line.strip().split()
            word2vec[line[0]] = np.array(line[1:], dtype=np.float64)

    return word2vec


def initial(words, word2vec):
    sentence = []
    zero = [0 for i in range(vec_len)]
    ones = [1 for i in range(vec_len)]
    num = 0
    for word in words:
        if num == sentence_len:
            break
        if word in word2vec.keys():
            sentence.append(word2vec[word])
        else:
            sentence.append(ones)
        num += 1
    for i in range(sentence_len - num):
        sentence.insert(0, zero)
    return sentence


def get_dict():
    dict = {}
    with open("trainfinal_en.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            lines = line.strip().split()
            for word in lines:
                if dict.get(word) == None:
                    dict[word] = cnt
                    cnt = cnt + 1
    return dict

def get_id_dict():
    dict = {}
    id_dict = {}
    with open("trainfinal_en.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            lines = line.strip().split()
            for word in lines:
                if dict.get(word) == None:
                    dict[word] = cnt
                    id_dict[cnt] = word
                    cnt = cnt + 1
    return id_dict

def get_cue_dict():
    cue_dict = {}
    with open("dict.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            words = line.strip().split()
            if len(words[0]) < 3:
                continue
            if (cnt == 999):
                break
            cue_dict[words[0]] = cnt
            cnt = cnt + 1
        cue_dict["EPT"] = 999
    return cue_dict

def get_cue_dict_id():
    cue_dict = {}
    with open("dict.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            words = line.strip().split()
            if len(words[0]) < 3:
                continue
            if (cnt == 999):
                break
            cue_dict[cnt] = words[0]
            cnt = cnt + 1
        cue_dict[999] = "EPT"
    return cue_dict


def load_data_cue_word(file):
    cue_dict = get_cue_dict()
    dict = get_dict()
    word2vec = read_word2vec("word2vec.txt")
    data = []
    target = []
    sentence_target = []
    with open(file, 'r', encoding='UTF-8') as f:
        lastline = None
        for line in f:
            if len(line) == 1:
                lastline = None
                continue
            if lastline != None:
                words = lastline.strip().split()
                sentence = initial(words[:-1], word2vec)
                data.append(sentence)
                words = line.strip().split()
                p = []
                for word in words[:-1]:
                    p.append(dict[word])
                while len(p) < sentence_len:
                    p.append(0)
                target.append(cue_dict[words[-1]])
                sentence_target.append(p[:sentence_len])
            lastline = line
    return np.array(data), np.array(target), np.array(sentence_target)


def load_data(file):
    cue_dict = get_cue_dict()
    dict = get_dict()
    word2vec = read_word2vec("word2vec.txt")
    with open(file, 'r', encoding='UTF-8') as f:
        data = []
        target = []
        for line in f:
            if len(line) == 1:
                words = line.strip().split()
                sentence = initial(words[:-1], word2vec)
                data.append(sentence)
                target.append(-1)
            else:
                words = line.strip().split()
                sentence = initial(words[:-1], word2vec)
                data.append(sentence)
                target.append(cue_dict[words[-1]])
    target.append(-1)
    return np.array(data), np.array(target)


if __name__ == '__main__':
    load_data_cue_word("trainfinal.txt")