import pandas as pd
dataset =  pd.read_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project3_datasets\\train.csv')
dataset_test = pd.read_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project3_datasets\\test.csv')
data = ['LotArea','YearBuilt','1stFlrSF','2ndFlrSF','HalfBath','BedroomAbvGr','TotRmsAbvGrd','OverallCond','MSSubClass']
##from sklearn.preprocessing import Imputer
##my_imptr = Imputer()
##data[:,8] = my_imptr.fit_transform(data[:,8])
train_x = dataset[data]
train_y = dataset.SalePrice
test_x = dataset_test[data]
from sklearn import linear_model
clf2 = linear_model.LogisticRegression()
prediction2 = clf2.fit(train_x, train_y).predict(test_x)
from sklearn.metrics import mean_absolute_error, accuracy_score
print "Mean Error -> ",mean_absolute_error(train_y.head(1459), prediction2)
print "Linear_Regression -> ",accuracy_score(train_y.head(1459), prediction2)
##from sklearn.model_selection import train_test_split
##train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2)
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier()
prediction1 = clf.fit(train_x, train_y).predict(test_x)
print "Decision Tree -> ",accuracy_score(train_y.head(1459), prediction1)
from sklearn.naive_bayes  import GaussianNB
clf = GaussianNB()
prediction3 = clf.fit(train_x, train_y).predict(test_x)
print "Naive Bayes -> ",accuracy_score(train_y.head(1459), prediction3)
##from sklearn import svm
##clf = svm.SVC()
##prediction4 = clf.fit(train_x, train_y).predict(test_x)
##print "SVM -> ",accuracy_score(train_y.head(1459), prediction4)
submission = pd.DataFrame({'Id':dataset_test.Id, 'SalePrice':prediction2})
submission.to_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project3_datasets\\Validated_dataset\\submission2.csv', index=False)
submission = pd.DataFrame({'Id':dataset_test.Id, 'SalePrice':prediction1})
submission.to_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project3_datasets\\Validated_dataset\\submission1.csv', index=False)
submission = pd.DataFrame({'Id':dataset_test.Id, 'SalePrice':prediction3})
submission.to_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project3_datasets\\Validated_dataset\\submission3.csv', index=False)
