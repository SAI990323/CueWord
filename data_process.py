import jieba.posseg
import re
import nltk.stem as ns

def read_en_train_data(file):
    '''
    进行分词处理 首先去除标点符号 然后按照要求 先将名词动词形容词还原到原型 然后统计出现次数 输出到dict.txt中
    :param file: 进行过remove处理的数据
    :return: 将去除标点以后的文件输出到train_en_new.txt
    '''
    import nltk
    dict = {}
    lemmatizer = ns.WordNetLemmatizer()
    with open(file, 'r', encoding='utf-8') as f:
        ff = open("train_en_new.txt", 'w', encoding='utf-8')
        for line in f:
            line = line.strip()

            if len(line) == 0:
                ff.write("\n")
                continue
            r = re.compile("[\.\!\/_,$%^*(+\"\']+|[+——！-；–「」》:：“”·‘’《，。？、~@#￥%……&*（）()?]+")
            # 以下两行过滤出中文及字符串以外的其他符号
            line = r.sub(' ', line)
            words = line.strip().split()
            words = nltk.pos_tag(words)
            for word_tuple in words:
                flag = False
                word = word_tuple[0]
                if word_tuple[1] == "NN" or word_tuple[1] == "NNS" or word_tuple[1] == "NNP" :
                    word = lemmatizer.lemmatize(word, pos = 'n')
                    flag = True
                if word_tuple[1] == "JJ" or word_tuple[1] == "JJR" or word_tuple[1] == "JJS":
                    word = lemmatizer.lemmatize(word, pos = 'a')
                    flag = True
                if word_tuple[1] == "VB" or word_tuple[1] == "VBD" or word_tuple[1] == "VBG" or word_tuple[1] == "VBN" or\
                        word_tuple[1] == "VBZ" or word_tuple[1] == "VBP":
                    word = lemmatizer.lemmatize(word, pos = 'v')
                    flag = True
                if flag:
                    if dict.get(word) == None:
                        dict[word] = 1
                    else :
                        dict[word] = dict[word] + 1
                ff.write(word + " ")
            ff.write("\n")
        ff.close()
    p = open('dict.txt', 'w', encoding='UTF-8')
    print("load finished!")
    l = sorted(dict.items(), key = lambda dict:-dict[1])
    for i in l:
        p.write(i[0] + " ")
        p.write(str(i[1]))
        p.write("\n")
    p.close()

def read_train_data(file):
    '''
    中文处理时使用这个 与read_en_train_data相同
    :param file:
    :return:
    '''
    dict = {}
    with open(file, 'r', encoding='UTF-8') as f:
        ff = open("trainnew.txt", 'w', encoding='UTF-8')
        for line in f:
            r = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+")
            # 以下两行过滤出中文及字符串以外的其他符号
            line = r.sub('', line)
            if len(line) == 0:
                ff.write("\n")
                continue
            words = jieba.posseg.lcut(line, HMM=True)
            for word in words:
                ff.write(word.word + " ")
                if word.flag == 'n' or word.flag == 'v' or word.flag == 'a':
                    if dict.get(word.word) == None:
                        dict[word.word] = 1
                    else:
                        dict[word.word] = dict[word.word] + 1
            ff.write("\n")
        ff.close()


    p = open('dict.txt', 'w', encoding='UTF-8')
    print("load finished!")
    l = sorted(dict.items(), key = lambda dict:-dict[1])
    for i in l:
        p.write(i[0] + " ")
        p.write(str(i[1]))
        p.write("\n")
    p.close()


def remove(file):
    '''
    删除小于三轮的对话
    :param file: train_en.txt 经过处理换行符后的文件
    :return: none
    '''
    with open(file, 'r', encoding = 'UTF-8') as f:
        ff = open('train_remove.txt', 'w', encoding='UTF-8')
        lines = []
        for line in f:
            r = re.compile("[\s+\.\!\/_,$%^*(+\"\']+|[+——！；「」》:：“”·‘’《，。？、~@#￥%……&*（）()]+")
            # 以下两行过滤出中文及字符串以外的其他符号
            line = r.sub('', line)
            lines.append(line)
            if len(line) == 0:
                if (len(lines) >= 2):
                    for l in lines:
                        ff.write(l + "\n")
                lines.clear()
        if (len(lines) >= 2):
            for l in lines:
                ff.write(l)

        ff.close()


reply = {}

def check(lastline, nowline):
    '''
    :param lastline:  询问
    :param nowline:  回答
    :return: 检查是否回答单词书>=3 且回答出现次数小于等于10次
    '''
    if len(lastline) >= 3 and lastline[0] == lastline[1] and lastline[1] == lastline[2]:
        return False
    if len(nowline) >= 3 and nowline[0] == nowline[1] and nowline[1] == nowline[2]:
        return False
    if nowline == lastline:
        return False
    if reply.get(nowline) == None:
        reply[nowline] = 1
    else :
        reply[nowline] = reply[nowline] + 1
    if (reply[nowline] > 10):
        return  False
    return True



def find_cue_word(dict, cue_dict, line, EPT = True):
    '''
    给定一个句话 将单词表中出现次数少于6次的单词化成UNK 并找到其中的cue_dict
    :param dict:  dict
    :param cue_dict: cue_dict
    :param line: 一句话
    :param EPT: 是否能够使用EPT作为cue word
    :return: 新的对话 和 cue_word 以及是否使用EPT 要保证EPT使用次数小于等于1000次
    '''
    lines = line.strip().split()
    list = []
    cue_word = None
    mx = 0
    for word in lines:
        if dict.get(word) < 6:
            list.append("UNK")
        else:
            list.append(word)
        if cue_dict.get(word) != None:
            if cue_dict[word] > mx:
                mx = cue_dict[word]
                cue_word = word
    if cue_word == None and EPT:
        return list, "EPT", 1
    if cue_word == None:
        return list, "NO", 1
    return list, cue_word, 0

def signal(file):
    '''
    首先预处理得到 dict 和 cue_dict 然后对每一句话 标注cue_dict 未标注成功的对话 删除
    :param file: train_en_new.txt
    :return: 输出到trainfinal_en.txt
    '''
    dict = {}
    cue_dict = {}
    with open("dict.txt", 'r', encoding='UTF-8') as f:
        cnt = 0
        for line in f:
            lines = line.strip().split()
            value = int(lines[1])
            if len(lines[0]) < 3:
                continue
            cnt = cnt + 1
            if cnt < 1000:
                cue_dict[lines[0]] = value

    with open(file,'r', encoding='UTF-8') as f:
        for line in f:
            lines = line.strip().split()
            for word in lines:
                if dict.get(word) == None:
                    dict[word] = 1
                else :
                    dict[word] = dict[word] + 1

    print("dict finished")
    ff = open('trainfinal_en.txt', 'w', encoding='UTF-8')
    with open(file, 'r', encoding='UTF-8') as f:
        label = 0
        lines = []
        for line in f:
            if len(line) == 1:
                if label < 1000:
                    for line in lines:
                        line1, cue_word1, t1 = find_cue_word(dict, cue_dict, line = line)
                        label += t1
                        for word in line1:
                            ff.write(word + " ")
                        ff.write(cue_word1 + "\n")
                    ff.write("\n")
                else :
                    flag =True
                    session = []
                    for line in lines:
                        line1, cue_word1, t1 = find_cue_word(dict, cue_dict, line = line, EPT=False)
                        if cue_word1 != "NO":
                            line1.append(cue_word1)
                        else:
                            flag = False
                        session.append(line1)
                    if flag:
                        for line in session:
                            for word in line:
                                ff.write(word + " ")
                            ff.write("\n")
                        ff.write("\n")
                lines.clear()
                continue
            lines.append(line)


def initial_file(file):
    '''
    对数据进行预处里 将数据中的__eou__ 换成换行 输出到train_en.txt
    :param file: 初始数据文件
    :return:  none
    '''
    with open(file, 'r', encoding='UTF-8') as f:
        ff = open("train_en.txt", 'w', encoding='utf-8')
        for line in f:
            words = line.strip().split()
            for word in words:
                if word == '__eou__':
                    ff.write("\n")
                    continue
                ff.write(word + " ")
            ff.write("\n")

if __name__ == '__main__':
    read_en_train_data("train_en.txt")
    signal("train_en_new.txt")
