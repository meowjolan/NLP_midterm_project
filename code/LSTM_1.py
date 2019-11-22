############################################
# Coding: utf-8
# Author: 郭俊楠
# Time: 2019-11-21
# Description： LSTM模型（单向）
############################################

# 导入相关模块
import numpy as np
import os
import pickle
import copy
import torch
import matplotlib.pyplot as plt
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块


def make_dict(file_path='./words_data/word_dict.txt'):
    '''
        创建映射词典
        参数：
            file_path: 词典文件路径
        参数：
            word2num: 词语映射到序号
            num2word: 序号映射到词语
    '''
    # 构建词典，将一个词映射到一个序号，以及将一个序号对应一个词
    word2num = {}
    num2word = {}
    with open(file_path, 'r') as f:
        # 遍历每一行
        for line in f:
            # 去掉换行符并分割
            number, word = line.strip().split(' ')
            # 加入词典
            word2num[word] = int(number)
            num2word[int(number)] = word

    return word2num, num2word


# 定义模型
class LSTM(nn.Module):
    def __init__(self, embedding_dim, lstm_dim, h1_dim, h2_dim, dropout, dict_size, num_lstm_layers):
        '''
            初始化模型
            参数：
                embedding_dim: 词向量维度
                h1_dim: 全连接层1的输入维数
                h2_dim: 全连接层2的输入维数
                dropout: 丢弃比例
                if_bidirectional: 是否双向
                dict_size: 词典的大小
            返回值：
                None
        '''
        super(LSTM, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.dropout = dropout
        self.dict_size = dict_size
        self.num_lstm_layers = num_lstm_layers
        
        # 建立模型
        self._build_model()
        
        
    def _build_model(self):
        '''
            建立模型
            参数：
                None
            返回值：
                None
        '''
        # 词嵌入层 - LSTM - 全连接层X2
        self.word_embeddings = nn.Embedding(self.dict_size+1, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.h1_dim, self.num_lstm_layers, batch_first=True, dropout=self.dropout)
        self.h1 = nn.Linear(self.h1_dim, self.h2_dim)
        self.h2 = nn.Linear(self.h2_dim, self.dict_size+1)
 
    def forward(self, batch_data):
        '''
            前向传播方法
            参数：
                batch_data: 一批句子
            返回值：
                output: 模型输出
        '''
        # 排序并填充句子
        # (batch_size, _)  ->  (batch_size, batch_len[0])
        batch_len = torch.tensor([len(i) for i in batch_data])
        for i in range(len(batch_data)):
            batch_data[i] += [0 for _ in range(batch_len[0]-len(batch_data[i]))]
        batch_data = autograd.Variable(torch.tensor(batch_data))
    
        # 嵌入层
        # (batch_size, batch_len[0])  ->  (batch_size, batch_len[0]， embedding_dim)
        batch_word_vector = self.word_embeddings(batch_data)
        
        # 打包
        batch_pack = torch.nn.utils.rnn.pack_padded_sequence(batch_word_vector, batch_len, batch_first=True)
        
        # LSTM层
        # (batch_size, batch_len[0]， embedding_dim) -> (batch_size, batch_len[0], h1_dim)
        lstm_output, _ = self.lstm(batch_pack)
        
        # 解包
        batch_pad, batch_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_output, batch_first=True)
        
        # 减少维数
        # (batch_size, batch_len[0], h1_dim)  ->  (_, h1_dim)
        h1_input = []
        for i in range(batch_pad.shape[0]):
            h1_input += batch_pad[i, :batch_len[i], :].data.numpy().tolist()
        h1_input = autograd.Variable(torch.tensor(h1_input))
        
        # 全连接层1-RELU激活
        # (_, h1_dim) -> (_, h2_dim)
        h1_output = self.h1(h1_input)
        h2_input = F.relu(h1_output)
        
        # 全连接层2-Softmax
        # (_, h2_dim) -> (_, dict_size)
        h2_output = self.h2(h2_input)
        output = F.log_softmax(h2_output, dim=1)
        
        return output


def make_train_set(file_path, word2num, batch_size):
    '''
        创建训练集
        参数：
            file_path: 训练集路径
            word2num: 词典
            batch_size: 一个批次的大小
        返回值：
            batch_data: 分批次的句子
            batch_target: 目标输出
    '''
    batch_data = [[]]
    cur_batch_size = 0
    
    # 选取1000个文本依次进行统计
    for i in range(1, 1001):
        # 打开文件
        with open(file_path+'{}.txt'.format(i), 'r') as f:
            # 遍历每一行（即一个句子的分词）
            for line in f:
                # 获得词列表
                words_list = line.strip().split(' ')
                # 将词列表映射为序号列表
                number_list = [word2num[word] for word in words_list]
                # 加入训练集
                batch_data[-1].append(number_list)
                cur_batch_size += 1
                if cur_batch_size == batch_size:
                    batch_data.append([])
                    cur_batch_size = 0
    
    # 构建目标输出
    batch_target = []
    for i in range(len(batch_data)):
        # 排序
        batch_data[i].sort(key=lambda x: len(x), reverse=True)
        
        batch_target.append([])
        for j in range(len(batch_data[i])):
            # 忽略句首，句尾补0
            batch_target[-1] += batch_data[i][j][1:] + [0]
    batch_target = [torch.tensor(batch_target[i]) for i in range(len(batch_target))]
        
    return batch_data, batch_target


# 构建测试集
def make_test_set(s_file, pos_file, word2num):
    '''
        创建测试集
        参数：
            s_file: 储存句子列表的文件
            pos_file: 储存要猜测的词的位置的文件
            word2num: 词典
        返回值：
            sentences: 句子列表
            targets: 目标输出
    '''
    with open(s_file, 'rb') as f:
        s_list = pickle.load(f)
    with open(pos_file, 'rb') as f:
        ans_pos = pickle.load(f)

    s_list = [s.split() for s in s_list]
    judge_func = lambda x : 0 if x not in word2num else word2num[x]
    sentences = [[s_list[i][j] for j in range(ans_pos[i]+1)] for i in range(len(s_list))]
    sentences = [[judge_func(s_list[i][j]) for j in range(ans_pos[i]+1)] for i in range(len(s_list))]
    
    # 进行排序
    sentences.sort(key=lambda x: len(x), reverse=True)
    ans_pos.sort(reverse=True)

    targets = [s[-1] for s in sentences]
    sentences = [s[:-1] for s in sentences]

    return sentences, torch.tensor(targets)



def train(batch_data, batch_target, batch_size, lstm_dim, h1_dim, h2_dim, \
          lr, epoch_num, dropout, num_lstm_layers, embedding_dim, dict_size):
    '''
        训练模型
        参数：
            batch_data: 训练集数据 
            batch_target: 目标单词(文本)
            其余: 超参数
        返回值：
            model: 训练过的模型
    '''
    # 模型实例化
    model = LSTM(embedding_dim, lstm_dim, h1_dim, h2_dim, dropout, dict_size, num_lstm_layers)
    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # 损失函数
    loss_func = nn.NLLLoss()
    # 损失值记录
    loss_record = []

    # 进行训练
    for epoch in range(epoch_num):
        # 确定数据批次
        batch_no = epoch % len(batch_data)
        # 模型输出
        output = model(copy.deepcopy(batch_data[batch_no]))
        # 计算损失
        loss = loss_func(output, batch_target[batch_no])
        # 反向传播，更新参数
        optimizer.zero_grad()           
        loss.backward()
        optimizer.step()
        # 输出并记录每一步的损失值
        loss_record.append(float(loss))
        if (epoch+1)%10 == 0:
            print('Epoch:  {0}/{1} | train loss: {2}'.format(epoch+1, epoch_num, loss))

    # 绘制图像
    plt.plot(loss_record,"r-",linewidth=1)   #在当前绘图对象绘图（X轴，Y轴，蓝色虚线，线宽度）
    plt.xlabel("Epoch") #X轴标签
    plt.ylabel("Loss")  #Y轴标签
    plt.show()  #显示图

    return model


def evaluate(batch_data, batch_target, sentences, targets):
    '''
        在训练集和测试集上评价模型
        参数：
            batch_data: 训练集数据 
            batch_target: 目标单词(文本)
            sentences: 测试集句子
            targets： 目标单词
        返回值：
            None
    '''
    # 训练集上的准确率
    total_count = 0
    correct_count = 0
    for i in range(len(batch_data)):
        # 取得一系列概率
        output = model(copy.deepcopy(batch_data[i]))
        # 取概率最大坐标
        predictions = output.argmax(dim=1)
        # 更新总数
        total_count += len(batch_target[i])
        # 更新正确计数
        correct_count += torch.sum(predictions == batch_target[i])
        
    print("Total: {}, Correct: {}, Accuracy: {}".format(total_count, correct_count, int(correct_count)/int(total_count)))

    # 在测试集上测试准确率
    correct_count = 0
    # 取得一系列概率
    output = model(copy.deepcopy(sentences))
    # 删去填充序号0的概率
    output = output[:, 1:]
    # 取概率最大坐标
    predictions = output.argmax(dim=1)+1
    # 计算每个句子末尾相应的位置
    pos_list = np.cumsum([len(s)-1 for s in sentences])
    predictions = torch.tensor([predictions[p] for p in pos_list])
    # 更新正确计数
    correct_count += torch.sum(predictions == targets)
        
    print("Total: {}, Correct: {}, Accuracy: {}".format(len(predictions), correct_count, int(correct_count)/len(predictions)))


if __name__ == "__main__":
    # 定义超参数
    # 一批数据包含多少个句子
    batch_size = 400
    # lstm层节点数
    lstm_dim = 256
    # 全连接层节点数
    h1_dim = 256
    h2_dim = 512
    # 学习率
    lr = 1e-3
    # 训练次数
    epoch_num = 6000
    # 丢弃比例
    dropout = 0.2
    # lstm层数
    num_lstm_layers = 2
    # 词向量维度
    embedding_dim = 300

    # 创建映射词典
    word2num, num2word = make_dict()
    # 构建训练集
    batch_data, batch_target = make_train_set('./words_data/', word2num, batch_size)
    # 训练模型
    model = train(batch_data, batch_target, batch_size, lstm_dim, h1_dim, h2_dim, \
                  lr, epoch_num, dropout, num_lstm_layers, embedding_dim, len(word2num))
    # 构建测试集
    sentences, targets = make_test_set("./final_s", "./ans_pos", word2num)
    # 模型评价
    evaluate(batch_data, batch_target, sentences, targets)