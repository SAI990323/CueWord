import torch
import torch.nn as nn
import load
from torch.utils import data
import numpy as np
import math


dict_size = 70

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

    def __init__(self, input_size = 100, hidden_size = 1000):
        self.hidden_size = hidden_size
        self.input_size = input_size
        super(CueWordSelectNet, self).__init__()
        self.encoder= nn.LSTM(input_size,hidden_size,2, batch_first=True, bias=False)
        self.layer1 = nn.Linear(hidden_size + 1000,4000)
        self.layer2 = nn.Linear(4000,1000)
        self.layer3 = nn.Linear(1000, hidden_size, bias=True)

    def forward(self, input, h):
        h_t = torch.zeros(2, input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        c_t = torch.zeros(2, input.size(0), self.hidden_size, dtype=input.dtype).to(device)
        output_sentence = torch.zeros(input.size(0), 44, 70, dtype=input.dtype).to(device)
        output, (h_t,c_t) = self.encoder(input, (h_t,c_t)) #encoder
        topic_tracker = torch.zeros(input.size(0), self.hidden_size).to(device).scatter_(1, h, 1) #need to add
        MLP_input = torch.cat((h_t[0],topic_tracker), dim = 1)
        MLP_output = self.layer1(MLP_input)
        MLP_output = self.layer2(MLP_output)

        return torch.sigmoid(MLP_output), (h_t, c_t)

class Decoder(nn.Module):
    def __init__(self, input_size = 100, hidden_size = 1000):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.layer = nn.Linear(hidden_size, hidden_size, bias=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, 2, batch_first=True, bias=False)
        self.linear = nn.Linear(hidden_size, dict_size)

    def forward(self, input, h):
        (h_t, c_t) = h
        input = self.layer(input)
        decoder_input = torch.zeros(input.size(0), 44, 1000).to(device)
        for i in range(44):
            decoder_input[:,i,:] = input
        output_sentence = torch.zeros(input.size(0), 44, 70, dtype=input.dtype).to(device)
        decoder_output, (h_t,c_t) = self.decoder(decoder_input, (h_t, c_t))
        for i, data in enumerate(decoder_output.chunk(input.size(1), dim = 1)):
            data = data.to(device)
            data = data.squeeze(1)
            output_sentence[:,i,:] = self.linear(data)
        return output_sentence




def train_cue_word(epochs = 100, batch_size = 64, learning_rate = 0.0001):
    data, target, sentence_target = load.load_data_cue_word("trainfinal.txt")
    trainset = MyData(data[:800], target[:800], sentence_target[:800])
    valiset = MyData(data[800:900], target[800:900], sentence_target[:800])
    testset = MyData(data[:-100], target[:-100], sentence_target[:800])
    trainset = torch.utils.data.DataLoader(trainset, batch_size=batch_size)
    valiset = torch.utils.data.DataLoader(valiset, batch_size=batch_size)
    testset = torch.utils.data.DataLoader(testset, batch_size=batch_size)
    net = CueWordSelectNet().to(device)
    dnet = Decoder().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    lossfunc = nn.CrossEntropyLoss()
    lastcorrect = 0
    for epoch in range(epochs):
        for data, target, sentence_target in trainset:
            input = data.to(device)
            target = target.to(device)
            sentence_target = sentence_target.to(device)
            cue_word , (h_t, c_t) = net(input, target.reshape(target.size(0), 1))
            mx = torch.argmax(cue_word, dim = 1).to(device).reshape(cue_word.size(0), 1)
            p = torch.zeros((input.size(0), 1000)).to(device).scatter_(1, mx, 1)
            decoder_input = torch.zeros(input.size(0), input.size(1), 1000).to(device)
            for i in range(input.size(1)):
                decoder_input[:, 0, :] = p.detach()
            sentences = dnet(decoder_input, (h_t, c_t))
            Loss = lossfunc(cue_word , target)
            for i, da in enumerate(sentences.chunk(sentences.size(1), dim=1)):
                da = da.to(device)
                da = da.squeeze(1)
                #tar = sentence_target[i].long().to(device)
                tar = torch.LongTensor(torch.zeros((batch_size)).long()).to(device)
                Loss = Loss + lossfunc(da, tar)
            net.zero_grad()
            Loss.backward()
            optimizer.step()
        with torch.no_grad():
            correct = 0
            total = 0
            for data, target, sentence_target in valiset:
                data = data.to(device)
                target = target.to(device)
                sentence_target = sentence_target.to(device)
                cue_word, sentences = net(input, target.reshape(target.size(0), 1))
                predicted = torch.argmax(cue_word)
                correct += (predicted == target).sum().item()
                total += data.size(0)
            print("epoch %d , valicorredt %.4f" % (epoch, correct / total))
            if (correct < lastcorrect + 10):
                learning_rate /= 2
                optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
            lastcorrect = correct
    with torch.no_grad():
        correct = 0
        total = 0
        for data, target, sentence_target in testset:
            data = data.to(device)
            target = target.to(device)
            sentence_target = sentence_target.to(device)
            cue_word, sentences = net(input, target.reshape(target.size(0), 1))
            predicted = torch.argmax(cue_word)
            correct += (predicted == target).sum().item()
            total += data.size(0)
    print("test correct %.4f" % (correct / total))
    state = {'net': net.state_dict()}
    torch.save(state, "single.model")



def train(epochs = 100, batch_size = 64, learning_rate = 0.0001):
    sampling_times = 5
    T = 3
    data, target = load.load_data("trainfinal.txt")
    word2vec = {}
    net = CueWordSelectNet().to(device)
    dnet = Decoder().to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    lines = []
    targets = []
    gamma = 0.9
    for epoch in range(epochs):
        for i in range(len(target)):
            if target[i] == -1:
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
                    compare = inputs
                    if j > 1:
                        compare = torch.cat([torch.Tensor(lines[j - 2]).reshape(inputs.size()).to(device), compare], dim = 0)
                    print(compare.size())
                    avg_rsk = 0
                    risk_list = []
                    for m in range(sampling_times):
                        p = torch.zeros(1,1000)
                        p[0, pred[0, m]] = 1
                        risk = 0
                        #cue_word = word2vec[pred[m]]
                        relevance = compare
                        ### risk1
                        cue_word1 = torch.randn(100, 1).to(device)
                        risk_sum = 0
                        for k in range(j, j + T):
                            sentence = dnet(p.to(device), (h_t, c_t))
                            mx = 1.9
                            for word in lines[j - 1]:
                                word = torch.Tensor(word.reshape(word.size, 1)).to(device)
                                mx = max(mx, torch.cosine_similarity(cue_word1, word, dim = 0))
                            mmx = 1.0
                            for word in lines[j]:
                                word = torch.Tensor(word.reshape(word.size, 1)).to(device)
                                mx = max(mx, torch.cosine_similarity(cue_word1, word, dim = 0))
                            risk = risk + 0.1 * torch.log(torch.mul(mx ,mmx))
                        ### risk2
                            # risk = risk + 0.9 * model(sentence.detach(), relevance.detach())
                            #
                            pp = torch.rand(1, 44,100).to(device)
                            relevance = torch.cat([relevance, pp], dim = 0)
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
                lines.clear()
                targets.clear()
            lines.append(data[i])
            targets.append(target[i])


if __name__ == '__main__':
    train()


