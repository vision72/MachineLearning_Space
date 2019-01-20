import pandas as pd
from sklearn.svm import SVC
train_X = pd.read_csv("D:\\WorkSpace\\MachineLearning_Space\\Basic_ML_Projects\\~Datasets\\~vision_datasets\\Project5_datasets\\train_test_dataset\\train.csv")
test_X = pd.read_csv("D:\\WorkSpace\\MachineLearning_Space\\Basic_ML_Projects\\~Datasets\\~vision_datasets\\Project5_datasets\\train_test_dataset\\test.csv")
train_y = pd.read_csv("D:\\WorkSpace\\MachineLearning_Space\\Basic_ML_Projects\\~Datasets\\~vision_datasets\\Project5_datasets\\train_test_dataset\\trainLabels.csv")
clf = SVC().fit(train_X, train_y)
pred = clf.predict(test_X)
var = len(test_X.index)
x = []
for y in range(var):
	x.append(y+1)
mySub = pd.DataFrame({'Id': x,'Solution':pred})
mySub.to_csv('submission.csv', index=False)