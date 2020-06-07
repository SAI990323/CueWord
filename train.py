import torch
import torch.nn as nn
import load
from torch.utils import data
import numpy as np
import math


dict_size = 4811

class MyData(data.Dataset):
    def __init__(self, data, target, sentence_target):
        self.data = data
        self.target = target
        self.sentence_target = sentence_target
        self.sentence_target = torch.LongTensor(self.sentence_target)
        self.target = torch.LongTensor(self.target)
        self.data = torch.FloatTensor(self.data)
    def __getitem__(self, index):
        data, target, sentence_target = self.data[index], self.target[index], self.sentence_target[index]
        return data, target, sentence_target

    def __len__(self):
        return len(self.data)

device = torch.device("cuda")

class CueWordSelectNet(nn.Module):

    def __init__(self, input_size = 600, hidden_size = 1000):
        self.hidden_size = hidden_size
        self.input_size = input_size
        super(CueWordSelectNet, self).__init__()
        self.encoder= nn.LSTM(input_size,hidden_size,2, batch_first=True, bias=False)
        self.layer1 = nn.Linear(hidden_size + 1000,4000)
        self.layer2 = nn.Linear(4000,1000)
        self.layer3 = nn.Linear(1000, hidden_size, bias=True)

    def forward(self, input, h, hh = None):
        if hh == None:
            h_t = torch.zeros(2, input.size(0), self.hidden_size, dtype=input.dtype).to(device)
            c_t = torch.zeros(2, input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        else :
            (h_t, c_t) = h
        output_sentence = torch.zeros(input.size(0), 44, dict_size, dtype=input.dtype).to(device)
        output, (h_t,c_t) = self.encoder(input, (h_t,c_t)) #encoder
        topic_tracker = torch.zeros(input.size(0), self.hidden_size).to(device).scatter_(1, h, 1) #need to add
        MLP_input = torch.cat((h_t[0],topic_tracker), dim = 1)
        MLP_output = self.layer1(MLP_input)
        MLP_output = self.layer2(MLP_output)

        return torch.softmax(MLP_output, dim = 0), (h_t, c_t)

class Decoder(nn.Module):
    def __init__(self, input_size = 600, hidden_size = 1000):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.decoder1 = nn.LSTMCell(hidden_size + hidden_size, hidden_size, bias=True)
        self.decoder2 = nn.LSTMCell(dict_size+ hidden_size + hidden_size, hidden_size, bias=True)
        self.linear = nn.Linear(hidden_size, dict_size)

    def forward(self, input, h):
        (h_list, c_list) = h
        h_t = h_list[0]
        h_t2 = h_list[1]
        c_t = c_list[0]
        c_t2 = c_list[1]
        input = self.layer(input)
        decoder_input = input
        output_sentence = torch.zeros(input.size(0), 44, dict_size, dtype=input.dtype).to(device)
        #decoder_output, (h_t,c_t) = self.decoder(decoder_input, (h_t, c_t))
        zeros = torch.zeros(input.size(0), dict_size, dtype=input.dtype).to(device)
        for i, data in enumerate(decoder_input.chunk(decoder_input.size(1), dim = 1)):
            data = data.to(device)
            data = data.squeeze(1)
            l1_input = torch.cat([data, h_t], dim = 1)
            (h_t, c_t) = self.decoder1(l1_input,(h_t, c_t))
            l2_input = torch.cat([zeros, h_t2, h_t], dim = 1)
            (h_t2, c_t2) = self.decoder2(l2_input, (h_t2,c_t2))
            output_sentence[:,i,:] = torch.softmax(self.linear(h_t2), dim = 0)
            zeros = output_sentence[:,i,:]
        return output_sentence



def test_cue_word():
    data, target, sentence_target = load.load_data_cue_word("trainfinal_en.txt")
    total = len(data)
    train_size = int(total / 10) * 8
    vali_size = int(total / 10) * 9
    testset = MyData(data[vali_size:], target[vali_size:], sentence_target[vali_size:])
    testset = torch.utils.data.DataLoader(testset, batch_size=64)
    checkpoint = torch.load("single_net.model")
    net = CueWordSelectNet().to(device)
    net.load_state_dict(checkpoint['net'])
    checkpoint = torch.load("final_dnet.model")
    dnet = Decoder().to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target, sentence_target in testset:
            input = data.to(device)
            target = target.to(device)
            sentence_target = sentence_target.to(device)
            cue_word, (h_t, c_t) = net(input, target.reshape(target.size(0), 1))
            predicted = torch.argmax(cue_word, dim=1)
            correct += (predicted == target).sum().item()
            total += data.size(0)
        print("test correct %.4f" % (correct / total))


def test(net = None, dnet = None):
    with torch.no_grad():
        sampling_times = 5
        T = 3
        #data, target = load.load_data("trainfinal_en.txt")
        id2dict = load.get_id_dict()
        id_dict = load.get_cue_dict_id()
        cue_dict = load.get_cue_dict()
        word2vec = load.read_word2vec("word2vec.txt")
        if net == None:
            checkpoint = torch.load("single_net.model")
            net = CueWordSelectNet().to(device)
            net.load_state_dict(checkpoint['net'])
        if dnet == None:
            checkpoint = torch.load("single_dnet.model")
            dnet = Decoder().to(device)
            dnet.load_state_dict(checkpoint['dnet'])
        with open("test.txt", 'r', encoding='UTF-8') as f:
            data = []
            target = []
            for line in f:
                if len(line) == 1:
                    lines = np.array(data)
                    targets = np.array(target)
                    L = len(lines)
                    for j in range(1, L):
                        inputs = np.array(lines[j - 1])
                        Cue = np.array(targets[j - 1])
                        inputs = np.expand_dims(inputs, axis=0)
                        Cue = np.expand_dims(Cue, axis=0)
                        Cue = Cue.reshape((1, 1))
                        inputs = torch.FloatTensor(inputs).to(device)
                        Cue = torch.LongTensor(Cue).to(device)
                        cue_word, (h_t, c_t) = net(inputs, Cue)
                        _, pred = torch.topk(cue_word, sampling_times)
                        for m in range(sampling_times):
                            p = torch.zeros(1, 1000)
                            p[0, pred[0, m]] = 1
                            p = p.unsqueeze(dim=0)
                            for k in range(j, j + T):
                                sentence = dnet(p.to(device), (h_t, c_t))
                                q = sentence.squeeze(dim = 0)
                                for t in range(len(q)):
                                    print(id2dict[int(torch.argmax(q[t]))])
                    data.clear()
                    target.clear()
                else:
                    words = line.strip().split()
                    sentence = load.initial(words[:-1], word2vec)
                    data.append(sentence)
                    target.append(cue_dict[words[-1]])

def train_cue_word(epochs = 10, batch_size = 64, learning_rate = 0.0001):
    data, target, sentence_target = load.load_data_cue_word("trainfinal_en.txt")
    total = len(data)
    train_size = int(total / 10) * 8
    vali_size = int(total / 10) * 9
    trainset = MyData(data[:train_size], target[:train_size], sentence_target[:train_size])
    valiset = MyData(data[train_size:vali_size], target[train_size:vali_size], sentence_target[train_size:vali_size])
    testset = MyData(data[vali_size:], target[vali_size:], sentence_target[vali_size:])
    trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    valiset = torch.utils.data.DataLoader(valiset, batch_size=batch_size)
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    net = CueWordSelectNet().to(device)
    dnet = Decoder().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer1 = torch.optim.Adam(dnet.parameters(), lr=learning_rate)
    lossfunc = nn.CrossEntropyLoss()
    lastcorrect = 0
    for epoch in range(epochs):
        for data, target, sentence_target in trainset:
            input = data.to(device)
            target = target.to(device)
            sentence_target = sentence_target.to(device)
            cue_word , (h_t, c_t) = net(input, target.reshape(target.size(0), 1))
            mx = torch.argmax(cue_word.detach(), dim = 1).to(device).reshape(cue_word.size(0), 1)
            p = torch.zeros((input.size(0), 1000)).to(device).scatter_(1, mx, 1)
            decoder_input = torch.zeros(input.size(0), input.size(1), 1000).to(device)
            for i in range(input.size(1)):
                decoder_input[:, 0, :] = p.detach()
            sentences = dnet(decoder_input.detach(), (h_t.detach(), c_t.detach()))
            # for i, da in enumerate(sentences.chunk(sentences.size(1), dim=1)):
            #     da = da.to(device)
            #     da = da.squeeze(1)
            #     tar = sentence_target[:,i].long().to(device)
            #     Loss = Loss + lossfunc(da, tar)
            da = sentences.reshape((sentences.size(0) * sentences.size(1), sentences.size(2)))
            tar = sentence_target.reshape((sentence_target.size(0) * sentence_target.size(1))).long()
            Loss = lossfunc(da, tar)
            dnet.zero_grad()
            Loss.backward()
            optimizer1.step()
            Loss = lossfunc(cue_word , target)
            net.zero_grad()
            Loss.backward()
            optimizer.step()

        with torch.no_grad():
            correct = 0
            total = 0
            for data, target, sentence_target in valiset:
                input = data.to(device)
                target = target.to(device)
                sentence_target = sentence_target.to(device)
                cue_word, (h_t, c_t) = net(input, target.reshape(target.size(0), 1))
                predicted = torch.argmax(cue_word, dim = 1)
                correct += (predicted == target).sum().item()
                total += data.size(0)
            print("epoch %d , valicorredt %.4f" % (epoch, correct / total))
            if (correct < lastcorrect + 10):
                learning_rate /= 2
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            lastcorrect = correct
        test(net, dnet)
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target, sentence_target in testset:
            input = data.to(device)
            target = target.to(device)
            sentence_target = sentence_target.to(device)
            cue_word, (h_t, c_t) = net(input, target.reshape(target.size(0), 1))
            predicted = torch.argmax(cue_word, dim=1)
            correct += (predicted == target).sum().item()
            total += data.size(0)
    print("test correct %.4f" % (correct / total))
    state = {'net': net.state_dict()}
    torch.save(state, "single_net.model")
    state = {'dnet': dnet.state_dict()}
    torch.save(state, "single_dnet.model")



def train(epochs = 100, batch_size = 64, learning_rate = 0.0001):
    sampling_times = 5
    T = 3
    #data, target = load.load_data("trainfinal_en.txt")
    id_dict = load.get_cue_dict_id()
    cue_dict = load.get_cue_dict()
    word2vec = load.read_word2vec("word2vec.txt")
    checkpoint = torch.load("single_net.model")
    net = CueWordSelectNet().to(device)
    net.load_state_dict(checkpoint['net'])
    checkpoint = torch.load("single_dnet.model")
    dnet = Decoder().to(device)
    dnet.load_state_dict(checkpoint['dnet'])
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    optimizer1 = torch.optim.Adam(dnet.parameters(), lr = learning_rate)
    lines = []
    targets = []
    gamma = 0.9

    with open("trainfinal_en.txt", 'r', encoding='UTF-8') as f:
        data = []
        target = []
        for line in f:
            if len(line) == 1:
                lines = np.array(data)
                targets = np.array(target)
                L = len(lines)
                for j in range(1, L):
                    inputs = np.array(lines[j - 1])
                    Cue = np.array(targets[j - 1])
                    inputs = np.expand_dims(inputs, axis=0)
                    Cue = np.expand_dims(Cue, axis=0)
                    Cue = Cue.reshape((1,1))
                    inputs = torch.FloatTensor(inputs).to(device)
                    Cue = torch.LongTensor(Cue).to(device)
                    cue_word, (h_t, c_t) = net(inputs, Cue)
                    _, pred = torch.topk(cue_word, sampling_times)
                    # compare = inputs
                    # if j > 1:
                    #     compare = torch.cat([torch.Tensor(lines[j - 2]).reshape(inputs.size()).to(device), compare], dim = 0)
                    # avg_rsk = 0
                    risk_list = []
                    for m in range(sampling_times):
                        p = torch.zeros(1,1000)
                        p[0, pred[0, m]] = 1
                        risk = 0
                        #cue_word = word2vec[pred[m]]
                        #relevance = compare
                        ### risk1
                        cue_word1 = torch.FloatTensor(word2vec[id_dict[int(pred[0, m])]]).to(device).reshape((600, 1))
                        risk_sum = 0
                        avg_rsk = 0
                        p = p.unsqueeze(dim = 0)
                        for k in range(j, j + T):
                            sentence = dnet(p.to(device), (h_t, c_t))
                            mx = 0.000001
                            for word in lines[j - 1]:
                                word = torch.Tensor(word.reshape(word.size, 1)).to(device)
                                mx = max(mx, torch.cosine_similarity(cue_word1, word, dim = 0))
                            mmx = 0.000001
                            for word in lines[j]:
                                word = torch.Tensor(word.reshape(word.size, 1)).to(device)
                                mx = max(mx, torch.cosine_similarity(cue_word1, word, dim = 0))
                            risk = risk - torch.log(torch.mul(mx ,mmx))
                        ### risk2
                            # risk = risk + 0.9 * model(sentence.detach(), relevance.detach())
                            #
                            #pp = torch.rand(1, 44,100).to(device)
                            #relevance = torch.cat([relevance, pp], dim = 0)
                            risk_sum = risk_sum + risk * math.pow(gamma, k - j)
                            avg_rsk = avg_rsk + risk_sum
                            risk_list.append(risk_sum)
                    avg_rsk = avg_rsk / sampling_times
                    Loss = 0
                    for m in range(sampling_times):
                        conv = risk_list[m] - avg_rsk
                        conv.requires_grad = False
                        Loss = Loss + torch.log(cue_word[0, pred[0,m]]) * conv
                    dnet.zero_grad()
                    net.zero_grad()
                    Loss.backward()
                    optimizer.step()
                data.clear()
                target.clear()
            else:
                words = line.strip().split()
                sentence = load.initial(words[:-1], word2vec)
                data.append(sentence)
                target.append(cue_dict[words[-1]])

    state = {'net': net.state_dict()}
    torch.save(state, "final_net.model")
    state = {'dnet': dnet.state_dict()}
    torch.save(state, "final_dnet.model")


if __name__ == '__main__':
    train_cue_word()

