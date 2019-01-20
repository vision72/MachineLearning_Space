import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import accuracy_score
from sklearn import naive_bayes
from sklearn import tree
from sklearn import svm

print "Importing Libs Successfull"

# importing the dataset
train_dataset = pd.read_csv("D:\\WorkSpace\\MachineLearning_Space\\Datasets\\set-1\\Project6_datasets\\train_test_dataset\\train.csv",sep=',',na_values='.',header=None,iterator=True, chunksize=100)
train_dataset2 = pd.read_csv("D:\\WorkSpace\\MachineLearning_Space\\Datasets\\set-1\\Project6_datasets\\train_test_dataset\\train.csv")
# before actually loading the data, we use chunksize to gradually load the data in parts
# test_dataset = pd.read_csv("D:\\WorkSpace\\MachineLearning_Space\\Datasets\\set-1\\Project6_datasets\\train_test_dataset\\test.csv", chunksize=1000)

# train & test data
train_X = train_dataset
train_y = train_dataset2['target']

# test_X = test_dataset.loc[:,'48df886f9':'d2919256b']

print "Dataset Loaded Successfully"


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
train_X = pd.DataFrame(train_X)
ax.plot(train_X, train_y)
plt.show()

# LabelEncoder_X = LabelEncoder()
# #test_X = list(test_X) # don't pay attention here...
# test_X = LabelEncoder_X.fit_transform(test_X)

# print "Preprocessing Done!"

# # # moving towards identifying the best classifier, or regressor for this problem

# # # DecisionTrees
# clf1 = tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=2)
# clf1.fit(train_X, train_y)
# pred1 = clf1.predict(test_X)

# # # Naive_Bayes ..(A good Choice with text data)
# clf2 = naive_bayes.GaussianNB()
# clf2.fit(train_X, train_y)
# pred2 = clf2.predict(test_X)

# # # SVM ..(Best with AdaBoost)
# clf3 = svm.SVC(kernel='rbf', c=10000, gamma=10)
# clf3.fit(train_X, train_y)
# pred3 = clf3.predict(test_X)

# print "All Algorithms Predicted.."
# print "Accuracy as Follows: \n"

# # # printing the accuracy
# print "Decision Trees Accuracy -> ", accuracy_score(pred1, train_y)
# print "Naive Bayes Accuracy -> ", accuracy_score(pred2, train_y)
# print "SVM Accuracy -> ", accuracy_score(pred3, train_y)

