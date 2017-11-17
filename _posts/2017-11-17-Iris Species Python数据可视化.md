---

layout: post
categories: [Machine Learning]
tags: [Machine Learning, Python,  Kaggle]

---


`Iris Species`是`sklearn`里很常用而简单的一个分类数据集，数据除了id和label外，还有四个属性:

- sepal length (cm)
- sepal width (cm)
- petal length (cm)
- petal width (cm)

为了能在二维图像中展示，以下就暂时只有petal length和petal width两个属性进行展示。

```Python
import pandas as pd
import numpy as np
from sklearn import datasets

iris = datasets.load_iris()
# 从sklearn.datasets导入数据，格式是sklearn.unit形式的

X = iris.data[:, [2, 3]]
y = iris.target
iris_df = pd.DataFrame(iris.data[:, [2, 3]], 
	columns=iris.feature_names[2:])
	
# 读取iris的data和target存入X和y，存入pandas.DataFrame里
# 暂时只保留petal length和petal width两个属性

print(iris_df.head())
print('\n' + 'The unique labels in this data are ' + 
	str(np.unique(y)))
# 打印前5个数据查看内容

```

输出结果：

```
   petal length (cm)  petal width (cm)
0                1.4               0.2
1                1.4               0.2
2                1.3               0.2
3                1.5               0.2
4                1.4               0.2

The unique labels in this data are [0 1 2]

```

将完整的数据集以7：3的比例分割成训练集和测试集

```Python
X_train, X_test, y_train, y_test = 
	train_test_split(X, y, test_size=.3, random_state=0)

print('There are {} samples in the training set 
	and {} samples in the test set'.
	format(X_train.shape[0], X_test.shape[0]))
```

输出结果：

```
There are 105 samples in the training set 
	and 45 samples in the test set
```

在常见的数据预处理中，将数据标准化是很重要的，能把被处理的数据缩放到一个合适的范围，有助于之后算法的收敛和准确性。

```Python
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_train)
# 用StandardScaler来fit训练数据

X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)

print('After standardizing our features, 
	the first 5 rows of our data now look like this:\n')
print(pd.DataFrame(X_train_std, 
	columns=iris_df.columns).head())
	
# 查看标准化结果

```
输出结果

```
After standardizing our features, 
	the first 5 rows of our data now look like this:

   petal length (cm)  petal width (cm)
0          -0.182950         -0.291459
1           0.930661          0.737219
2           1.042022          1.637313
3           0.652258          0.351465
4           1.097702          0.737219
```

然后通过matplotlib.pyplot包来生成原始的分类图像

```Python
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt

markers = ('s', 'x', 'o')
# 设定标记类型
colors = ('red', 'blue', 'lightgreen')
# 设定颜色
cmap = ListedColormap(colors[:len(np.unique(y_test))])
for idx, cl in enumerate(np.unique(y)):
	plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1], c=cmap(idx), 
		marker=markers[idx], label=cl)
plt.show()
```

![](http://ope2etmx1.bkt.clouddn.com/Figure_1.png)

在正式应用模型之前，先定义一个之后展示的方法。

```Python
def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, 
	test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
    plt.show()
```

使用SVM模型

```Python
svm = SVC(kernel='rbf', random_state=0, gamma=.10, C=1.0)
# 用rbf核
svm.fit(X_train_std, y_train)

print('The accuracy of the svm classifier on training data is 
	{:.2f} out of 1'.format(svm.score(X_train_std, y_train)))
print('The accuracy of the svm classifier on test data is 
	{:.2f} out of 1'.format(svm.score(X_test_std, y_test)))
# 打印svm模型在训练集和测试集上的得分

plot_decision_regions(X_test_std, y_test, svm)
# 展示图像
```
输出结果：

```
The accuracy of the svm classifier 
	on training data is 0.95 out of 1
The accuracy of the svm classifier 
	on test data is 0.98 out of 1
```
![](http://ope2etmx1.bkt.clouddn.com/Figure_01.png)

使用KNN模型

```Python
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(X_train_std, y_train)

print('The accuracy of the knn classifier is 
	{:.2f} out of 1 on training data'.
	format(knn.score(X_train_std, y_train)))
print('The accuracy of the knn classifier is 
	{:.2f} out of 1 on test data'.
	format(knn.score(X_test_std, y_test)))

plot_decision_regions(X_test_std, y_test, knn)
```

输出结果：

```
The accuracy of the knn classifier 
	is 0.95 out of 1 on training data
The accuracy of the knn classifier 
	is 1.00 out of 1 on test data

```
![](http://ope2etmx1.bkt.clouddn.com/Figure_2.png)

使用XGB

```Python
xgb_clf = xgb.XGBClassifier()
xgb_clf = xgb_clf.fit(X_train_std, y_train)

print('The accuracy of the xgb classifier is 
	{:.2f} out of 1 on training data'.
	format(xgb_clf.score(X_train_std, y_train)))
print('The accuracy of the xgb classifier is 
	{:.2f} out of 1 on test data'.
	format(xgb_clf.score(X_test_std, y_test)))

plot_decision_regions(X_test_std, y_test, xgb_clf)
```


输出结果：

```
The accuracy of the xgb classifier 
	is 0.98 out of 1 on training data
The accuracy of the xgb classifier 
	is 0.98 out of 1 on test data
```
![](http://ope2etmx1.bkt.clouddn.com/Figure_3.png)

在测试集上，KNN表现最好，而在训练集上，SVM和XGB表现更加优越。