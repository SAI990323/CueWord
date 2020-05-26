import numpy as np

def initial(words, word2vec):
    sentence = []
    zero = [0 for i in range(100)]
    num = 0;
    for word in words:
        if num == 44:
            break
        if word in word2vec.keys():
            sentence.append(word2vec[word])
            num += 1
    for i in range(44 - num):
        sentence.insert(0, zero)
    return sentence

def get_dict():
    dict = {}
    with open("trainfinal.txt",'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            lines = line.strip().split()
            for word in lines:
                if dict.get(word) == None:
                    cnt= cnt + 1
                    dict[word] = cnt
    return dict

def get_cue_dict():
    cue_dict = {}
    with open("dict.txt", 'r' , encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            cnt = cnt + 1
            if (cnt == 1000):
                break
            words = line.strip().split()
            cue_dict[words[0]] = cnt
        cue_dict["EPT"] = 1000
    return cue_dict

def load_data_cue_word(file):
    cue_dict = get_cue_dict()
    dict = get_dict()
    word2vec = {}
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
                while len(p) < 44:
                    p.append(0)
                target.append(cue_dict[words[-1]])
                sentence_target.append(p[:44])
            lastline = line
            if len(data) == 1000:
                break
    return np.array(data), np.array(target), np.array(sentence_target)


def load_data(file):
    cue_dict = get_cue_dict()
    dict = get_dict()
    word2vec = {}
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
            if len(data) == 1000:
                break
    target.append(-1)
    print(np.shape(data))
    print(np.shape(target))
    return np.array(data), np.array(target)

if __name__ == '__main__':
    data, target = load_data("trainfinal.txt")
    print(np.shape(data))