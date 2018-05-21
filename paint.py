# -*- coding: utf-8 -*-
__author__ = 'Kay'

import matplotlib.pylab as plt
import matplotlib.mlab as mlab
import pandas as pd
import numpy as np
import seaborn as sns
sns.set(style='ticks', palette='muted')


def hist_extraversion():
    '''
    外倾性分数的分布, 及其正态分布曲线
    :return:
    '''
    n_bins = 10
    data = pd.read_csv('data/regress_train_data.txt', sep=' ', header=None)
    ext = data[1]
    mu = ext.mean()
    sigma = ext.std()
    print(mu, sigma)
    fig = plt.figure(figsize=(10, 8))
    # --- for *.eps --- #
    fig.set_rasterized(True)
    # plt.title("The distribution of score on extraversion")
    plt.xlabel("$Score\ on\ extraversion$", fontsize=20)
    plt.ylabel("$Probability$", fontsize=20)
    plt.grid(True)
    plt.hist(ext, n_bins, normed=1, alpha=0.8, rwidth=0.85)
    x = np.linspace(0, 60, 100)
    y = mlab.normpdf(x, mu, sigma)
    plt.xlim(0, 60)
    plt.ylim(0, 0.055)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.plot(x, y, 'r--')
    # plt.tight_layout()
    plt.savefig('figure/ext_dist.eps', dpi=300)
    plt.show()


def shopping_hist():
    n_bins = 100
    data1 = pd.read_csv('data/split_class/large_IGNORE_404_shopping_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_404_shopping_-1.txt', sep=' ', header=None)

    shopping1 = data1[2]
    shopping2 = data2[2]
    for i in np.arange(3, 17):
        shopping1 += data1[i]
        shopping2 += data2[i]

    col1 = shopping1 / data1[1]
    print(col1.describe())
    col2 = shopping2 / data2[1]
    print(col2.describe())

    plt.subplot(1, 2, 1)
    plt.hist(col1, n_bins, normed=True, stacked=True, alpha=0.8, color='r', linewidth=1.5)
    plt.xlim(0, 0.5)
    plt.ylabel("frequency")
    plt.subplot(1, 2, 2)
    plt.hist(col2, n_bins, normed=True, stacked=True, alpha=0.8, color='b', linewidth=1.5)
    plt.xlim(0, 0.5)
    plt.ylabel("frequency")

    # plt.hist(data1[1], n_bins, normed=1, alpha=0.6, color='b', cumulative=True)
    # plt.hist(data2[1], alpha=0.6, color='r')
    plt.show()


def badge_bar(index):
    data1 = pd.read_csv('data/split_class/large_IGNORE_404_badge_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_404_badge_-1.txt', sep=' ', header=None)

    col1 = data1[index]
    col2 = data2[index]

    print(col1.value_counts() / len(col1))
    print(col2.value_counts() / len(col2))
    plt.pie(col1.value_counts() / len(col1), colors=('b', 'r'), labels=['badge', 'non-badge'])
    plt.show()


def badge_hist(index):
    n_bins = 5
    data1 = pd.read_csv('data/split_class/large_IGNORE_404_badge_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_404_badge_-1.txt', sep=' ', header=None)

    col1 = data1[index]
    col2 = data2[index]
    print(col1.mean())
    # print(col1.describe())
    print(col2.mean())
    # print(col2.describe())
    plt.subplot(1, 2, 1)
    plt.hist(col1, n_bins, alpha=0.8, color='r', linewidth=1.5)
    # plt.xlim(0, 0.5)

    plt.ylabel("frequency")
    plt.subplot(1, 2, 2)
    plt.hist(col2, n_bins, alpha=0.8, color='b', linewidth=1.5)
    # plt.xlim(0, 0.5)
    # plt.ylabel("frequency")

    # plt.hist(data1[1], n_bins, normed=1, alpha=0.6, color='b', cumulative=True)
    # plt.hist(data2[1], alpha=0.6, color='r')
    # plt.show()

def mood_hist(index):
    n_bins = 10
    data1 = pd.read_csv('data/split_class/large_IGNORE_406_mood_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_406_mood_-1.txt', sep=' ', header=None)

    mood_sum1 = pd.Series([0] * data1.shape[0])
    mood_sum2 = pd.Series([0] * data2.shape[0])
    # for i in np.arange(1, 7):
    for i in np.arange(1, 6):
        print(i)
        mood_sum1 += data1[i]
        mood_sum2 += data2[i]

    col1 = data1[index] / mood_sum1
    col2 = data2[index] / mood_sum2
    print(col1, col2)
    print(col1.mean())
    # print(col1.describe())
    print(col2.mean())
    # print(col2.describe())
    plt.subplot(1, 2, 1)
    plt.hist(col1, n_bins, alpha=0.8, color='r', linewidth=1.5)
    # plt.xlim(0, 0.5)

    plt.ylabel("frequency")
    plt.subplot(1, 2, 2)
    plt.hist(col2, n_bins, alpha=0.8, color='b', linewidth=1.5)
    # plt.xlim(0, 0.5)
    # plt.ylabel("frequency")
    plt.show()


def shopping_cumulative_hist():
    '''
    购物累积分布
    :return:
    '''
    n_bins = 10000
    data1 = pd.read_csv('data/split_class/large_IGNORE_425_shopping_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_425_shopping_-1.txt', sep=' ', header=None)

    # shopping1 = data1[2]
    # shopping2 = data2[2]
    # for i in np.arange(3, 17):
    #     shopping1 += data1[i]
    #     shopping2 += data2[i]
    #
    # col1 = shopping1 / data1[1]
    # col2 = shopping2 / data2[1]

    col1 = data1[2] / data1[1]
    col2 = data2[2] / data2[1]
    col1 = col1[col1<=0.16]
    col2 = col2[col2<=0.16]
    # print(len(col1))
    # print(len(col2))
    plt.figure(figsize=(10, 8))
    plt.hist(col1.dropna(), n_bins, normed=1, alpha=0.9, color=(1, 0.4, 0.2), linewidth=3.0, histtype='step', cumulative=True)
    plt.hist(col2.dropna(), n_bins, normed=1, alpha=0.9, color=(0.2, 0.0, 0.9), linewidth=3.0, histtype='step', cumulative=True)
    plt.grid(True)
    plt.xlim(0, 0.15)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(np.linspace(0, 1, 11), fontsize=20)
    plt.xlabel("Purchasing Index", fontsize=24)
    plt.ylabel("Cumulative probability", fontsize=24)
    plt.legend(['Extroverts', 'Introverts'], loc=4, fontsize=20)
    plt.savefig('figure/purchase_pro.eps', dpi=300, facecolor='white')
    # plt.hist(data1[1], n_bins, normed=1, alpha=0.6, color='b', cumulative=True)
    # plt.hist(data2[1], alpha=0.6, color='r')
    plt.show()


def driving_cumulative_hist():
    '''
    驾驶累积分布
    :return:
    '''
    n_bins = 5000
    data = pd.read_csv('data/drive_index.txt', header=None)
    data1 = data[data[1] == 1]
    data2 = data[data[1] == 0]


    # shopping1 = data1[2]
    # shopping2 = data2[2]
    # for i in np.arange(3, 17):
    #     shopping1 += data1[i]
    #     shopping2 += data2[i]
    #
    # col1 = shopping1 / data1[1]
    # col2 = shopping2 / data2[1]

    col1 = data1[9]
    col2 = data2[9]
    print(col1.describe())
    print(col2.describe())

    col1 = col1[col1 <= 0.2]
    col2 = col2[col2 <= 0.2]
    # print(len(col1))
    # print(len(col2))
    # plt.figure(figsize=(5, 4))
    plt.hist(col1.dropna(), n_bins, normed=1, linewidth=1.5, histtype='step', cumulative=True)
    plt.hist(col2.dropna(), n_bins, normed=1, linewidth=1.5, histtype='step', cumulative=True)
    plt.grid(True)
    plt.xlim(0, 0.02)
    plt.ylim(0, 1)
    # plt.xticks(fontsize=20)
    plt.yticks(np.linspace(0, 1, 11))
    plt.xlabel("Drive Index")
    plt.ylabel("Cumulative probability")
    plt.legend(['Extroverts', 'Introverts'], loc=4)
    # plt.savefig('figure/purchase_pro.eps', dpi=300, facecolor='white')
    # plt.hist(data1[1], n_bins, normed=1, alpha=0.6, color='b', cumulative=True)
    # plt.hist(data2[1], alpha=0.6, color='r')
    plt.show()


def mood_cumulative_hist(index):
    n_bins = 10000
    data1 = pd.read_csv('data/split_class/large_IGNORE_406_mood_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_406_mood_-1.txt', sep=' ', header=None)

    mood_sum1 = pd.Series([0] * data1.shape[0])
    mood_sum2 = pd.Series([0] * data2.shape[0])
    # for i in np.arange(1, 7):
    for i in np.arange(1, 6):
        mood_sum1 += data1[i]
        mood_sum2 += data2[i]
        # print(mood_sum1[0])

    # print(data1[index][0], mood_sum1[0])
    col1 = data1[index] / mood_sum1
    col2 = data2[index] / mood_sum2


    # q1 = pd.qcut(col1, np.linspace(0, 1, 11), retbins=True)
    # q2 = pd.qcut(col2, np.linspace(0, 1, 11), retbins=True)
    # print('----------------------------------------------')
    # print(q1)
    # print(col1.describe())
    # print(q2)
    # print(col2.describe())
    plt.figure(figsize=(8, 6))
    # plt.hist(col1.dropna(), n_bins, normed=1, alpha=0.9, color=(1, 0.4, 0.2), linewidth=3.0, histtype='step', cumulative=True)
    # plt.hist(col2.dropna(), n_bins, normed=1, alpha=0.9, color=(0.2, 0.0, 0.9), linewidth=3.0, histtype='step', cumulative=True)
    plt.hist(col1.dropna(), n_bins, normed=1, alpha=0.9, color=(1, 0.4, 0.2), linewidth=3.0, cumulative=True)
    plt.hist(col2.dropna(), n_bins, normed=1, alpha=0.9, color=(0.2, 0.0, 0.9), linewidth=3.0, cumulative=True)
    plt.grid(True)
    if index == 2:
        plt.legend(['Extroverts', 'Introverts'], loc=4, fontsize=20)
    list_xlim = [0.5, 0.3, 0.8, 0.5, 0.3]
    plt.xlim(0, list_xlim[index - 1])
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(np.linspace(0, 1, 11), fontsize=20)

    # plt.xlabel("Fear index", fontsize=25)
    plt.ylabel("Cumulative probability", fontsize=25)
    list_mood = ['anger', 'disgust', 'joy', 'sadness', 'fear']
    # plt.savefig('figure/mood_%s.eps' % list_mood[index - 1], dpi=300, facecolor='white')
    plt.show()


def negative_mood_cumulative_hist():
    n_bins = 5000
    data1 = pd.read_csv('data/split_class/large_IGNORE_406_mood_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_406_mood_-1.txt', sep=' ', header=None)

    mood_sum1 = pd.Series([0] * data1.shape[0])
    mood_sum2 = pd.Series([0] * data2.shape[0])
    mood_neg1 = pd.Series([0] * data1.shape[0])
    mood_neg2 = pd.Series([0] * data2.shape[0])
    # for i in np.arange(1, 7):
    for i in np.arange(1, 6):
        mood_sum1 += data1[i]
        mood_sum2 += data2[i]
        # print(mood_sum1[0])

    for i in [1, 2, 4, 5]:
        mood_neg1 += data1[i]
        mood_neg2 += data2[i]
        # print(mood_sum1[0])

    # print(data1[index][0], mood_sum1[0])
    col1 = mood_neg1 / mood_sum1
    col2 = mood_neg2 / mood_sum2
    # print(col1, col2)
    # print(col1.mean())
    print(col1.describe())
    # print(col2.mean())
    print(col2.describe())
    plt.figure(figsize=(6.6, 6))
    plt.hist(col1.dropna(), n_bins, normed=1, alpha=0.7, color='r', linewidth=2.0, histtype='step', cumulative=True)
    plt.hist(col2.dropna(), n_bins, normed=1, alpha=0.7, color='b', linewidth=2.0, histtype='step', cumulative=True)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Fear index", fontsize=20)
    plt.ylabel("Cumulative probability", fontsize=20)
    plt.show()


def ca_box_plot_shopping():
    # 读取数据
    data1 = pd.read_csv('data/split_class/large_IGNORE_425_shopping_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_425_shopping_-1.txt', sep=' ', header=None)
    col1 = data1[2] / data1[1]
    col2 = data2[2] / data2[1]
    # print(col1.describe())
    # print(col2.describe())
    # col1.to_csv("shopping_+1.txt")
    # col2.to_csv("shopping_-1.txt")
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=[col1, col2], fliersize=0.1, width=0.3)
    # sns.violinplot(data=[col1, col2], fliersize=0.1, width=0.3)

    plt.xticks((0, 1), ('Extroverts', 'Introverts'), fontsize=20)
    # plt.xlim(0.5, 2.5)

    plt.yticks(fontsize=20)
    plt.ylabel("Purchasing Index", fontsize=20)
    plt.ylim(0, 0.12)

    # plt.boxplot(data=[col1, col2], vert=False, sym='k+', showmeans=True, showfliers=True, notch=1)
    # plt.yticks((1, 2), ('Extroverts', 'Introverts'), fontsize=25, rotation=30)
    # plt.ylim(0.5, 2.5)
    #
    # plt.xticks(fontsize=30)
    # plt.xlabel("Purchasing Index", fontsize=30)
    # plt.xlim(0, 0.12)
    plt.savefig('figure/purchase_box.eps', dpi=300)
    plt.show()


def ca_box_plot_driving():
    # 读取数据
    # n_bins = 5000
    data = pd.read_csv('data/drive_index.txt', header=None)
    data1 = data[data[1] == 0]
    data2 = data[data[1] == 1]
    col1 = data1[9]
    col2 = data2[9]
    col1 = col1[col1 <= 0.2]
    col2 = col2[col2 <= 0.2]
    # print(col1.describe())
    # print(col2.describe())
    # col1.to_csv("shopping_+1.txt")
    # col2.to_csv("shopping_-1.txt")
    plt.figure(figsize=(8, 4))
    sns.boxplot(data=[col1, col2], width=0.3)
    # sns.violinplot(data=[col1, col2], fliersize=0.1, width=0.3)

    plt.xticks((0, 1), ('Extroverts', 'Introverts'))
    # plt.xlim(0.5, 2.5)

    # plt.yticks(fontsize=20)
    plt.ylabel("Drive Index")
    plt.ylim(0, 0.015)

    # plt.boxplot(data=[col1, col2], vert=False, sym='k+', showmeans=True, showfliers=True, notch=1)
    # plt.yticks((1, 2), ('Extroverts', 'Introverts'), fontsize=25, rotation=30)
    # plt.ylim(0.5, 2.5)
    #
    # plt.xticks(fontsize=30)
    # plt.xlabel("Purchasing Index", fontsize=30)
    # plt.xlim(0, 0.12)
    # plt.savefig('figure/purchase_box.eps', dpi=300)
    plt.show()


def ca_box_plot_features(index):
    '''
    训练特征和外倾性的关系
    :param index: 训练特征编号, 印象笔记存储
    :return:
    '''
    index += 1
    data1 = pd.read_csv('data/split_class/large_IGNORE_404_NOR_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_404_NOR_-1.txt', sep=' ', header=None)

    col1 = data1[index]
    col2 = data2[index]

    plt.boxplot([col1, col2], showmeans=True, showfliers=False)
    plt.show()


def ca_hist_features(index):
    '''
    训练特征和外倾性的关系
    :param index: 训练特征编号, 印象笔记存储
    :return:
    '''
    index += 1
    data1 = pd.read_csv('data/split_class/large_IGNORE_404_NOR_+1.txt', sep=' ', header=None)
    data2 = pd.read_csv('data/split_class/large_IGNORE_404_NOR_-1.txt', sep=' ', header=None)

    col1 = data1[index]
    col2 = data2[index]

    plt.subplot(1, 2, 1)
    plt.suptitle("introversion")
    plt.hist(col1, alpha=0.6)
    plt.subplot(1, 2, 2)
    plt.suptitle("extraversion")
    plt.hist(col2, alpha=0.6)
    plt.show()


if __name__ == '__main__':

    # 心理测试数据的分布
    # hist_extraversion()

    # 购物
    # shopping_cumulative_hist()
    # shopping_hist()
    # ca_box_plot_shopping()

    # 训练特征
    # ca_hist_features(35)
    # ca_box_plot_features(35)

    # 徽章
    # badge_hist(4)
    # badge_bar(5)

    # 情绪累积分布
    # mood_cumulative_hist(1)
    # mood_cumulative_hist(2)
    # mood_cumulative_hist(3)
    # mood_cumulative_hist(4)
    # mood_cumulative_hist(5)

    # negative_mood_cumulative_hist()

    driving_cumulative_hist()
    # ca_box_plot_driving()
