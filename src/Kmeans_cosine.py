# -*- coding: utf-8 -*-
# @Time    : 2019/6/27 21:25
# @Author  : Weiyang
# @File    : Kmeans_cosine.py

########################################################################################################################
# 基于点与点之间的余弦相似度实现KMeans聚类算法

# 算法思想：
# 首先选择K个中心点，然后将每个样本点分配到与其余弦相似度最大的中心点(越相似的点在一起)所在的类别，之后更新每个簇的中心点；
# 如果相邻两次聚类 每个簇的中心点不变，则停止，否则循环下去；

# 算法流程：
# 1. 选定要聚类的类别数目K
# 2. 选择K个中心点
# 3. 针对每个样本点，找到与其余弦相似度最大的中心点；与同一中心点余弦相似度最大的点为一个类簇，这样便完成一次聚类
# 4. 判断聚类前后的样本点的类别情况是否相同，如果相同，则算法停止，否则进入step5
# 5. 针对每个类簇中的样本点，计算这些样本点的中心点，当做该簇的新中心点，继续step3

# 算法关键处：
# 1. 初始类簇的中心点选择：
#    1) 随机选择，效果不理想
#    2) 选择彼此最不相似(余弦值最小)的K个点作为初始中心点：
#       计算所有样本点之间的余弦相似度，选择最不相似的一个点对(两个样本a,b),从样本点集中删除这两个点；
#       从剩余点中计算每个点与已知聚类中心点的余弦相似度的最大值(比如，相对而言，某个点与a最相似，某个点与b最相似)，
#       然后排序余弦相似度，选择余弦相似度最小的那个点；
#       以此类推，直至选出满足K个初始类簇的中心点；

# 2. 中心点的个数K的选择：
#    1) 拐点法(手肘法)：
#       a. 画出不同K值的 类内距离/类间距离-K的曲线，值越小聚类结果越好，选择图中的拐点处；
#       b. 画出不同K值的 SSE-K(误差平方和)，值越小聚类结果越好，选择图中的拐点处；
#          SSE是误差平方和，它本是所有样本点到它所在聚类中心点的距离之和，
#          这里改为所有样本点与它所在聚类中心的余弦相似度之和的负值；
# 3. 相似度的度量：两个点之间的余弦相似度
########################################################################################################################

import numpy as np
import matplotlib.pyplot as plt
import random
from collections import defaultdict

class KMeans:
    '''KMeans聚类算法'''
    def __init__(self,n_clusters,init='k-means++',distance='Cosine',max_iter=1000):
        self.n_clusters = n_clusters # 聚类中心的个数,至少是2
        self.init = init # 初始聚类中心的选择方式,可选参数为: 'random'，随机选择；'k-means++',以更有效的方式选择
        self.distance = distance # 距离的度量方式,可选参数为: 'C': 曼哈顿距离；'E':欧式距离；'M':闵式距离
        self.max_iter = max_iter # 算法收敛之前的最大迭代次数
        self.init_clusters = None # 初始聚类中心
        self.init_clusters_index = None # 初始聚类中心在点集中的索引
        self.cluster = None # 最终的聚类中心
        self.clusters = None # 最终的每个聚类的点簇集合
        self.SSE = None # 误差平方和

    def getPointCosine(self,points1,points2):
        '''获取点集points1与points2中每两个点之间的余弦相似度,原理:Cos(A,B) = A*B/|A||B|'''
        A, B = np.array(points1[:]), np.array(points2[:])
        BT = np.transpose(B)
        A_BT = np.dot(A, BT)
        Asq = A ** 2
        Asq = np.tile(np.sum(Asq, axis=1, keepdims=True), (1, A_BT.shape[1]))
        Bsq = BT ** 2
        Bsq = np.tile(np.sum(Bsq, axis=0, keepdims=True), (A_BT.shape[0], 1))
        Cosine = A_BT / (np.sqrt(Asq) * np.sqrt(Bsq)) # 两个点集中每两个点的余弦相似度
        return Cosine

    def selectCenterPoints(self,points):
        '''选取初始聚类中心，points是样本点集'''
        if self.init == 'random':
            num = len(points) # 样本点的数量
            if num < self.n_clusters:
                print('选取的聚类中心点过多,请缩小聚类簇的个数')
                exit()
            centers = [points[i] for i in random.sample(range(0,num),self.n_clusters)] # 生成不重复的点
        elif self.init == 'k-means++':
            num = len(points)  # 样本点的数量
            if num < self.n_clusters:
                print('选取的聚类中心点过多,请缩小聚类簇的个数')
                exit()
            # 首先选取最不相似度的一个点对
            # 计算points中每两个点之间的余弦相似度
            distance = self.getPointCosine(points,points)
            # 最小余弦相似度的索引
            ind_max = np.unravel_index(np.argmin(distance),distance.shape)
            centers = [] # 存储聚类中心
            centers_index = [ind_max[0],ind_max[1]] # 存储聚类中心的索引
            remaining_points = set(range(0,len(points))) - set(centers_index) # 剩余点集
            # 将这两个点加入到聚类中心集中
            centers.extend([points[ind_max[0]],points[ind_max[1]]])
            # 循环添加点到聚类中心集
            while True:
                # 判断初始聚类中心点的数量是否达到要求
                if len(centers) == self.n_clusters:
                    break
                # 遍历每个剩余点,寻找当前点距离已知聚类中心点最近的距离
                index = None # 存储余弦相似度最小的索引
                dis = 1 # 存储余弦相似度的最小值
                for point in remaining_points:
                    # 当前点与聚类中心点的余弦相似度最大值
                    temp = max([distance[point,i] for i in centers_index])
                    if temp < dis :
                        dis = temp
                        index = point
                # 将当前点加入到聚类中心
                centers.append(points[index])
                centers_index.append(index)
                remaining_points = set(range(0,len(points))) - set(centers_index) # 剩余点集
        return centers,centers_index

    def cal_center_point(self,cluster):
        '''计算多个点的中心(即各个维度值取平均即可)，作为新的聚类中心 cluster是点簇'''
        # 点簇中点的个数
        n = len(cluster)
        m = np.array(cluster).transpose().tolist()
        new_center = [sum(x)/float(n) for x in m]
        return new_center # 点簇的新聚类中心

    def check_center_diff(self,center,new_center):
        '''检查旧聚类中心与新聚类中心是否有差别'''
        for c,nc in zip(center,new_center):
            if c != nc:
                return False
        return True

    def cal_SSE(self,clusters,centers):
        '''计算每个样本点到它所在聚类中心的余弦相似度之和,clusters是聚类簇点集，center是相应的聚类中心点集'''
        SSE = 0
        # 遍历类簇
        for label in clusters.keys():
            # 当前类簇的样本点集
            points = clusters[label]
            # 当前类簇的聚类中心
            center = [centers[label]]
            # 计算当前类簇中每个样本点与聚类中心的误差平方和,误差平方和由余弦相似度来度量
            distance = self.getPointCosine(points,center)
            # 求和累加
            SSE += np.sum(distance)
        return -SSE

    def fit_predict(self,points):
        '''计算聚类中心,并输出每个样本的聚类类别,points 是样本点集,格式为list列表'''
        # 获取初始聚类中心
        self.init_clusters,self.init_clusters_index = self.selectCenterPoints(points)
        # 初始聚类中心
        center = self.init_clusters[:]
        # 聚类的类别
        label_indexs = list(range(0,self.n_clusters))
        # 循环聚类
        count = 1 # 记录聚类迭代的次数
        while True:
            # 每个样本的类别
            labels = []
            # 每个簇包含的样本
            clusters = defaultdict(list)
            # 计算样本点集中每个点与聚类中心的余弦相似度
            distance = self.getPointCosine(points,center)
            # 遍历每个样本点，获取与其最相似的聚类中心
            for index in range(len(points)):
                label = np.argmax(distance[index])
                # 记录当前样本点的类别
                labels.append(label)
                # 将当前样本点加入到对应的簇中
                clusters[label].append(points[index])
            # 对每个类簇，我们需要将该簇的聚类中心加进来
            for label in label_indexs:
                clusters[label].append(center[label])

            # 新聚类中心点集
            new_center = []
            # 对每个点簇计算新的聚类中心
            for label in label_indexs:
                nc = self.cal_center_point(clusters[label])
                new_center.append(nc)

            # 判断聚类前后是否有差别
            flag = self.check_center_diff(center,new_center)
            # 无差别时，停止聚类
            if flag == False:
                break
            count += 1
            # 判断是否达到最大迭代次数
            if count > self.max_iter:
                break
            center = new_center[:] # 新聚类中心替换旧聚类中心

        self.SSE = self.cal_SSE(clusters,new_center) # 计算SSE误差平方和
        self.cluster = new_center # 最终的聚类中心
        self.clusters = clusters # 聚类的点簇集合

        return labels # 每个点的类别

    def plot_clusters(self,clusters):
        '''展示聚类结果: clusters聚类结果的点簇集合'''
        for label in clusters.keys():
            data = np.array(clusters[label]) # 当前簇的点集
            data_x = [x[0] for x in data]
            data_y = [x[1] for x in data]
            plt.scatter(data_x,data_y)
        plt.show()

    def predict(self,points):
        '''批量预测点集points的类别'''
        labels = [] # 每个样本点的类别
        # 计算每个样本点与所有聚类中心的余弦相似度
        distance = self.getPointCosine(points,self.cluster)
        # 遍历每个样本点，获取与其最相似的聚类中心的余弦相似度
        for index in range(len(points)):
            # 获取当前样本点的类别
            label = np.argmax(distance[index])
            # 记录当前样本点的类别
            labels.append(label)
        return labels

if __name__ == '__main__':
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
    #print(points)
    k = KMeans(5, distance='C')
    print(k.fit_predict(points))
    print(k.cluster)
    print(k.SSE)
    print(k.predict(points))
    k.plot_clusters(k.clusters) # 展示聚类结果