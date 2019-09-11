# -*- coding: utf-8 -*-
# @Time    : 2019/6/28 16:38
# @Author  : Weiyang
# @File    : SelectK.py

########################################################################################################################
# 用于确定KMeans算法的K值
# 算法原理：画出误差平方和SSE与K值的曲线，取曲线中拐点处的K值
########################################################################################################################

from Kmeans_cosine import KMeans
import matplotlib.pyplot as plt
import random

# 生成一个样本集，用于测试KMmeans算法
def get_test_data():
    N = 1000
    # 产生点的区域,共5个区域，会聚类成5个点簇
    area_1 = [0, N / 4, N / 4, N / 2]
    area_2 = [N / 2, 3 * N / 4, 0, N / 4]
    area_3 = [N / 4, N / 2, N / 2, 3 * N / 4]
    area_4 = [3 * N / 4, N, 3 * N / 4, N]
    area_5 = [3 * N / 4, N, N / 4, N / 2]
    areas = [area_1, area_2, area_3, area_4, area_5]
    # 在各个区域内，随机产生一些点
    points = []
    for area in areas:
        rnd_num_of_points = random.randint(50, 200)
        for r in range(0, rnd_num_of_points):
            rnd_add = random.randint(0, 100)
            rnd_x = random.randint(area[0] + rnd_add, area[1] - rnd_add)
            rnd_y = random.randint(area[2], area[3] - rnd_add)
            points.append([rnd_x, rnd_y])
    return points
# 生成测试集
points = get_test_data()
# 为确定聚类的K值，我们画出SSE-K曲线图
SSE = []
for i in range(2, 11):
    k = KMeans(i, distance='C')
    print(k.fit_predict(points))
    SSE.append(k.SSE)
plt.plot(SSE,c='r')
plt.show()