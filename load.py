import numpy as np

sentence_len = 44
vec_len = 600


# 按照论文所说 一句话最多44个词 不足补0 超过截断

def read_word2vec(file):
    '''
    :param file: 给定一个gensim训练以后输出到文件的word2vec文件 要求其格式为 单词 向量 \n
    :return: 返回一个map，key为word, value为vector 其中EPT定义为一个随机向量, UNK定义为全1向量 其中词向量长度为00
    '''
    with open(file, 'r', encoding='utf-8') as f:
        word2vec = {}
        cnt = 0
        for line in f:
            if cnt == 0:
                cnt = 1
                continue
            line = line.strip().split()
            word2vec[line[0]] = np.array(line[1:], dtype=np.float64)
    word2vec["EPT"] = np.random.rand(600)
    word2vec["UNK"] = np.ones(600)
    return word2vec


def initial(words, word2vec):
    '''
    :param words: 一个list, 对应了输入数据一行的所有单词
    :param word2vec: word2vec对应的map
    :return: 返回一个44 * 600的list，每一个位置对应一个词向量, 若words长度不足，则在前面补全0向量
    '''
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
    '''
    :return: 给每一个word标号，并返回一个map key为word， value为id
    '''
    dict = {}
    with open("trainfinal_en.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            lines = line.strip().split()
            for word in lines:
                if dict.get(word) == None:
                    cnt = cnt + 1
                    dict[word] = cnt
    dict["UNK"] = 0
    return dict

def get_id_dict():
    '''
    :return: 给每一个word标号返回一个map key为id， value为word
    '''
    dict = {}
    id_dict = {}
    with open("trainfinal_en.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            lines = line.strip().split()
            for word in lines:
                if dict.get(word) == None:
                    cnt = cnt + 1
                    dict[word] = cnt
                    id_dict[cnt] = word
    id_dict[0] = "UNK"
    return id_dict

def get_cue_dict():
    '''
    :return: 给每一个cue_word标号，并返回一个map key为cue_word， value为id
    '''
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
    '''
    :return:   给每一个cue_word标号返回一个map key为id， value为cue_word
    '''
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
    '''
    :param file: 处理好的训练文件
    :return data: 训练数据 每一个是一个query (n * 44 * 600)
    :return target：cue_word target 对应一个query anwser的cue_word n
    :return sentence_target: 对应一个query 训练集中的anwswer (n * 22 * 600)
    '''
    cue_dict = get_cue_dict()
    dict = get_dict()
    word2vec = read_word2vec("word2vec.txt")
    data = []
    target = []
    sentence_target = []
    mx = 0
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
                    mx = max(mx, dict[word])
                while len(p) < sentence_len // 2:
                    p.append(0)
                target.append(cue_dict[words[-1]])

                sentence_target.append(p[:sentence_len // 2])
            lastline = line
    return np.array(data), np.array(target), np.array(sentence_target)


def load_data(file):
    '''
    :param file:  处理好的训练文件
    :return data: 返回对话训练用的数据 先认为每轮对话有3句进行测试  n * 3 * 44 * 100
    :return target： 对应每句话的cue_word
    '''
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
    load_data_cue_word("trainfinal_en.txt")