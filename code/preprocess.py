############################################
# Coding: utf-8
# Author: 郭俊楠
# Time: 2019-10-18
# Description： 对爬取的新闻进行预处理，包括分词、
#               分句和去噪等。
############################################

import jieba
import re
import os
import pickle

def ifValid(s):
    '''
        判断一个字符串是否由中文（utf8）组成
        参数：
            s：待判断的字符串
        返回值：
            result：布尔值，判断结果
    '''
    for word in s:
        if not ('\u4e00' <= word <= '\u9fff'):
            # 发现不为中文的字或符号
            return False
    # 字符串合法
    return True


def textToWords(text):
    '''
        将一个文本进行分词处理，返回词语列表
        参数：
            text：文本字符串
        返回值：
            word_list：词语列表，每个句子的词语组成一个列表
    '''
    # 先进行分句
    sentences = re.split('。|！|\!|？|\?', text)
    # 使用jieba进行分词
    word_list = [jieba.lcut(s) for s in sentences if s != '']
    # 给分词结果去噪
    for i in range(len(word_list)):
        word_list[i] = [w for w in word_list[i] if ifValid(w)]
    
    return word_list


def dealWithAllText(max_num):
    '''
        处理所有文本，构建词典、转化成词列表并保存为文件
        参数：
            max_num：要处理的文本数量
        返回值：
            None
    '''
    # 词表
    record = {}
    # 词频统计
    word_count = {}
    # 遍历读取每一个文本文件
    for i in range(1, max_num+1):
        # 打开文件
        with open('./news/{}.txt'.format(i)) as f1:
            # 获取文本内容
            text = f1.read()
            # 获取分词结果
            word_list = textToWords(text)
            # 构建词表并统计词频
            for l in word_list:
                for w in l:
                    if w not in record:
                        record[w] = len(record)
                        word_count[w] = 1
                    else:
                        word_count[w] += 1
            # 保存结果
            with open('./words_data/{}.txt'.format(i), 'w') as f2:
                for l in word_list:
                    if len(l) > 0:
                        f2.write(' '.join(l) + '\n')
                        f2.flush()
    # 保存词典
    with open('./words_data/word_dict.txt', 'w') as f:
        for word, number in record.items():
            f.write(str(number) + ' ' + word + '\n')
            f.flush()
    print('Got a words dictionary with size {}'.format(len(record)))

    # 处理测试集
    # 前半个句子的列表
    s1_list = []
    # 后半个句子的列表
    s2_list = []
    # 先处理句子
    with open('questions.txt', 'r') as q_f:
        # 每一行都是一个句子，逐行处理
        for line in q_f.readlines():
            # 分成前后两部分
            s1, s2 = line.split('[MASK]')
            # 分别处理前后两部分
            s1_list.append(textToWords(s1)[0])
            if textToWords(s2):
                s2_list.append(textToWords(s2)[0])
            else:
                s2_list.append([])
    
    # 过滤训练集并获取过滤词表
    filter_words = word_filter(max_num, record, word_count)

    # 过滤测试集
    for i in range(len(s1)):
        s1_list[i] = [w for w in s1_list[i] if w not in filter_words]
        s2_list[i] = [w for w in s2_list[i] if w not in filter_words]

    # 记录要猜测的词的位置（从0开始计数）并保存
    ans_pos = [len(s) for s in s1_list]
    with open('ans_pos', 'wb') as f:
        pickle.dump(ans_pos, f)

    # 获取答案
    with open('answer.txt', 'r') as f:
        ans = [w for w in f.read().split()]

    # 将答案填入句子
    final_s = [' '.join(s1_list[i])+' '+ans[i]+' '+' '.join(s2_list[i]) for i in range(len(s1_list))]
    # 保存完整句子
    with open('final_s', 'wb') as f:
        pickle.dump(final_s, f)


def word_filter(max_num, word_dict, word_count, stop_word_path='stop_words.txt'):
    '''
        去除停止词和低频词
    '''
    # 构造过滤词列表
    filter_words = []
    # 加入低频词
    for word, count in word_count.items():
        if count < 2:
            filter_words.append(word)
    # 加入停止词
    with open(stop_word_path) as f:
        filter_words += f.read().split()
    
    # 记录当前保存的文件序号，以忽略空文件
    file_number = 1
    # 遍历读取每一个文本文件
    for i in range(1, max_num+1):
        words = []
        # 读取文本
        with open('./words_data/{}.txt'.format(i)) as f:
            # 读取每一行
            for line in f:
                # 过滤词
                words.append([word for word in line.split() if word not in filter_words])

        # 去除空句子
        words = [l for l in words if l]
        
        # 判断是否有句子
        if words:
            # 保存文本
            with open('./words_data/{}.txt'.format(file_number), 'w') as f:
                for l in words:
                    # 忽略长度太小的句子
                    if len(l) > 2:
                        f.write(' '.join(l) + '\n')
                        f.flush()

            file_number += 1
    
    # 删除多余的文件
    for i in range(file_number, max_num+1):
        os.remove('./words_data/{}.txt'.format(i)) 

    return filter_words


if __name__ == "__main__":
    dealWithAllText(1000)

