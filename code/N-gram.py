############################################
# Coding: utf-8
# Author: 郭俊楠
# Time: 2019-11-21
# Description： N-gram模型
############################################


# 导入相关模块
import numpy as np
import os
import pickle


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


def add_ternary_count(i, j, k, dict_size, ternary_count):
    '''
        统计ternary_count（注意：这里假定已经有了ternary_count字典变量）
        参数：
            i, j, k：分别是三个词的序号
            dict_size: 词典大小
            ternary_count: 三元统计
        返回值：
            None
    '''
    if (i, j) not in ternary_count:
        ternary_count[(i, j)] = np.zeros(dict_size+1)
    ternary_count[(i, j)][k] = ternary_count[(i, j)][k] + 1


def word_count(word2num, path='./words_data/'):
    '''
        统计训练集
        参数：
            word2num: 映射词典
            path: 文件目录
        返回值：
            unary_count: 一元统计
            binary_count: 二元统计
            ternary_count: 三元统计
    '''
    dict_size = len(word2num)
    # 对训练语料进行频数统计
    # （词i）出现的频数
    unary_count = np.zeros(dict_size+1, 'int32')
    # （词i，词j）出现的频数
    binary_count = np.zeros((dict_size+1, dict_size+1), 'int32')
    # （词i，词j,词k）出现的频数 （三维数组耗内存，改用字典形式）
    ternary_count = {}
    # 选取1000个文本依次进行统计
    for i in range(1, 1001):
        # 打开文件
        with open(path+'{}.txt'.format(i), 'r') as f:
            # 遍历每一行（即一个句子的分词）
            for line in f:
                # 获得词列表
                words_list = line.strip().split(' ')
                # 将词列表映射为序号列表
                number_list = [word2num[word] for word in words_list]
                # 统计三种分布情况的频数
                for j in range(len(number_list)):
                    unary_count[number_list[j]] += 1
                    if j == 0:
                        binary_count[0, number_list[0]] += 1
                        add_ternary_count(0, 0, number_list[0], dict_size, ternary_count)
                    else:
                        binary_count[number_list[j-1], number_list[j]] = \
                            binary_count[number_list[j-1], number_list[j]] + 1
                        if j == 1:
                            add_ternary_count(0, number_list[0], number_list[1], dict_size, ternary_count)
                        else:
                            add_ternary_count(number_list[j-2], number_list[j-1], number_list[j], dict_size, ternary_count)

    return unary_count, binary_count, ternary_count


def predict_backoff(word_list, unary_count, binary_count, ternary_count):
    '''
        采用回退策略进行预测并评价：
            使用N-gram模型，使N尽可能大，其中1<=N<=3。
        参数：
            word_list：上文单词序号列表（未知词序号为-1）
            unary_count：一元统计词频
            binary_count：二元统计词频
            ternary_count：三元统计词频
        返回值：
            predict_number：预测单词序号
    '''
    # 根据上文单词个数选用预测方式
    if len(word_list) == 0:
        # 2-gram
        predict_number = np.argmax(binary_count[0, :])
    elif len(word_list) == 1:
        if (0, word_list[0]) not in ternary_count:
            # 1-gram
            predict_number = np.argmax(unary_count)
        else:
            # 3-gram
            predict_number = np.argmax(ternary_count[(0, word_list[0])])
    else:
        if (word_list[-2], word_list[-1]) in ternary_count:
            # 3-gram
            predict_number = np.argmax(ternary_count[(word_list[-2], word_list[-1])])
        elif word_list[-1] != -1 and np.max(binary_count[word_list[-1], :]) > 0:
            # 2-gram
            predict_number = np.argmax(binary_count[word_list[-1], :])
        else:
            # 1-gram
            predict_number = np.argmax(unary_count)
            
    return predict_number


def evaluate_on_training_set(word2num, unary_count, binary_count, ternary_count, path='./words_data/'):
    '''
        使用训练集进行评价
        参数：
            word2num: 映射词典
            unary_count：一元统计词频
            binary_count：二元统计词频
            ternary_count：三元统计词频
            path: 训练集目录
        返回值：
            None
    '''
    # 总计数
    total_count = 0
    # 正确计数
    correct_count = 0

    # 选取1000个文本依次进行统计
    for i in range(1, 1001):
        # 打开文件
        with open(path+'{}.txt'.format(i), 'r') as f:
            # 遍历每一行（即一个句子的分词）
            for line in f:
                # 获得词列表
                word_list = line.strip().split(' ')
                # 将词列表映射为序号列表
                number_list = [word2num[word] for word in word_list]
                for word in word_list:
                    if word in word2num:
                        number_list.append(word2num[word])
                    else:
                        number_list.append(-1)
                # 更新总计数
                total_count += len(number_list)-1
                # 预测并计数（忽略第一个词）
                for j in range(1, len(number_list)):
                    if number_list != -1:
                        # 无法正确预测未知的词
                        predict_number = predict_backoff(number_list[:j], unary_count, binary_count, ternary_count)
                        if predict_number == number_list[j]:
                            correct_count += 1

    print('On training set:')
    print("Total: {}, Correct: {}, Accuracy: {}".format(total_count, correct_count, correct_count/total_count))


def evaluate_on_test_set(word2num, unary_count, binary_count, ternary_count, path='./'):
    '''
        使用测试集进行评价
        参数：
            word2num: 映射词典
            unary_count：一元统计词频
            binary_count：二元统计词频
            ternary_count：三元统计词频
            path: 训练集目录
        返回值：
            predicitons: 预测结果
    '''
    # 读取测试集
    with open(path+'final_s', 'rb') as f:
        s_list = pickle.load(f)
    with open(path+'ans_pos', 'rb') as f:
        ans_pos = pickle.load(f)

    # 将句子列表映射为相应的序号列表
    number_list = []
    for s in s_list:
        number_list.append([])
        for w in s.split():
            if w in word2num:
                number_list[-1].append(word2num[w])
            else:
                number_list[-1].append(-1)
    
    # 正确计数
    correct_count = 0
    predictions = []
    # 逐句测试
    for i in range(len(s_list)):
        predictions.append(predict_backoff(number_list[i][:ans_pos[i]], unary_count, binary_count, ternary_count))
        if predictions[-1] == number_list[i][ans_pos[i]]:
            correct_count += 1

    print('On test set:')
    print("Total: {}, Correct: {}, Accuracy: {}".format(len(s_list), correct_count, correct_count/len(s_list)))

    return predictions


if __name__ == "__main__":
    # 创建映射词典
    word2num, num2word = make_dict()
    # 统计词频
    unary_count, binary_count, ternary_count = word_count(word2num)
    # 训练集上评价
    evaluate_on_training_set(word2num, unary_count, binary_count, ternary_count)
    # 测试集上评价
    predictions = evaluate_on_test_set(word2num, unary_count, binary_count, ternary_count)
    predictions = [num2word[i] for i in predictions]
    # 保存预测结果
    with open('./prediction_NGram.txt', 'w') as f:
        f.write('\n'.join(predictions))