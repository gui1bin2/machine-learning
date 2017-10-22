# -*- coding: utf-8 -*-

import numpy

from sklearn import datasets
#引入数据集
iris = datasets.load_iris()

iris

#查看数据的规模
iris.data.shape
#查看训练目标的总类
numpy.unique(iris.target)

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    iris.data, 
    iris.target, 
    test_size=0.3
)

data_train.shape
data_test.shape
target_train.shape
target_test.shape

from sklearn import neighbors

knnModel = neighbors.KNeighborsClassifier(n_neighbors=3)

knnModel.fit(data_train, target_train) 

knnModel.score(data_test, target_test)


from sklearn.model_selection import cross_val_score

cross_val_score(
    knnModel, 
    iris.data, iris.target, cv=5
)

#使用模型进行预测
knnModel.predict([[0.1, 0.2, 0.3, 0.4]])
