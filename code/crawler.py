############################################
# Coding: utf-8
# Author: 郭俊楠
# Time: 2019-10-17
# Description： 爬虫程序，抓取新浪科技频道滚动新闻
############################################

# 相关模块
import requests
from bs4 import BeautifulSoup
import selenium.webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
import re
import time
import random


def getNewsList(num):
    '''
        爬取滚动新闻页面，获取新闻标题及链接列表
        参数：
            num：要爬取的页数
        返回值：
            news_titles：新闻标题及链接的列表，大小为(num*50)
    '''
    # 新浪科技频道滚动新闻链接
    url = 'https://tech.sina.com.cn/roll/rollnews.shtml#pageid=372&lid=2431&k=&num=50&page={}'
    # drive设置及实例化
    chrome_options = Options()
    chrome_options.add_argument('--headless')
    driver = selenium.webdriver.Chrome(chrome_options=chrome_options)
    # 返回的列表
    news_titles = []
    # 遍历各个页面，爬取新闻标题及链接
    for i in range(1, num+1):
        # 获取页面内容
        driver.get(url.format(i))
        # 获取主页文本内容
        page_txt = driver.page_source
        # 解析页面文本内容
        soup = BeautifulSoup(page_txt, 'lxml')
        # 爬取新闻标题，加入列表
        news_titles = news_titles + soup.select('#d_list > ul > li > span.c_tit > a')
    # 关闭driver
    driver.close()

    return news_titles


def textFilter(text):
    '''
        获取新闻正文，并过滤html标签等无效信息
        参数：
            text：页面内容
        返回值：
            news_text：过滤后的新闻正文
    '''
    # 获取正文内容
    news_text = re.findall(r'<!--新增众测推广文案end-->.*<!-- <div class="show_statement">', 
                        text, flags=re.DOTALL)
    # 忽略不符合搜索模式的页面
    if (len(news_text) == 0):
        return None
    # 过滤标签
    news_text = re.sub(r'\<.*?\>', '',news_text[0], flags= re.DOTALL)
    # 替代集
    html_tag = {'&#xA;': ' ', '&quot;': '\"', '&amp;': '', '&lt;': '<', '&gt;': '>',
                '&apos;': "'", '&nbsp;': ' ', '&yen;': '¥', '&copy;': '©', '&divide;': '÷', 
                '&times;': 'x', '&trade;': '™', '&reg;': '®', '&sect;': '§', '&euro;': '€',
                '&pound;': '£', '&cent;': '￠', '&raquo;': '»', '\u3000': ' ', '\n': ' '}
    # 替代正文中的每一个符号
    for key, value in html_tag.items():
        news_text = news_text.replace(key, value)
    # 去掉空格
    news_text = re.sub(r'\ +', '', news_text)
    
    return news_text


def getNews(num):
    '''
        获取新闻正文，并保存为文件
        参数：
            num：最多获取(num*50)条新闻
        参数：
            valid_num：实际获取并保存的有效新闻数
    '''
    # 获取新闻链接
    news_titles = getNewsList(num)
    # 处理每一条新闻页面
    valid_num = 0
    for title in news_titles:
        # 获取新闻页面内容
        news_page = requests.get(title.get('href'))
        # 更改编码
        news_page.encoding = 'utf8'
        # 获取新闻正文
        news_text = textFilter(news_page.text)
        # 判断正文是否有效
        if news_text != None:
            # 更新记录
            valid_num = valid_num + 1
            # 保存至文件
            with open('./news/{}.txt'.format(valid_num), 'w') as f:
                f.write(news_text)
                f.flush()

    return valid_num


if __name__ == "__main__":
    # 至多爬取(num*50)条新闻
    num = 40
    # 开始爬取
    valid_num = getNews(num)
    print('Got {} piece(s) of technology news'.format(valid_num))
