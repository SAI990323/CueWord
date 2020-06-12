# CueWord
load.py 用来从处理好的数据中读入数据
data_process.py 用来预处理数据 分词 去除脏数据 处理出现次数很少的单词
word2vec_train 训练词向量
train.py 核心训练程序 基本完成 
train_cue_word 第一轮训练cue_word
train 第二轮训练对话 (目前因为参数原因 以及 论文中SMN模型无法使用 所以暂时效果不好 已经禁用)

环境：
gensim + nlt + pytorch1.5 + torchvision 1.1

运行方法：
python data_process.py
python word2vec_train.py
python train.py