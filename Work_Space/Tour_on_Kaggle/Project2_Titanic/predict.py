from sklearn.metrics import confusion_matrix, mean_absolute_error, accuracy_score
from sklearn.preprocessing import Imputer
import pandas as pd
import numpy as np
titanic_training_dataset = pd.read_csv("C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project2_datasets\\train.csv")
train_y = titanic_training_dataset['Survived']
titanic_test_dataset = pd.read_csv("C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project2_datasets\\test.csv")
imputer = Imputer()
psng = titanic_test_dataset['PassengerId']
sel_num_data = ['PassengerId', 'Age', 'Pclass', 'SibSp', 'Parch']
train = imputer.fit_transform(titanic_training_dataset[sel_num_data])
##from sklearn.preprocessing import LabelEncoder
##label = LabelEncoder()
##train[:,2] = label.fit_transform(train[:,2])
test = imputer.fit_transform(titanic_test_dataset[sel_num_data]) 
train_x = train
test_x = test
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=12, criterion='entropy',random_state=0)
clf.fit(train_x, train_y)
prediction = clf.predict(test_x)
##submission = pd.DataFrame({'PassengerId': psng,'Survived': prediction})
##submission.to_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project2_datasets\\Validated_dataset\\submission.csv', index=False)
train_y = titanic_training_dataset['Survived'][:418]
cm = confusion_matrix(train_y, prediction)
print("mean error ->",mean_absolute_error(train_y, prediction))
print("accuracy score ->",accuracy_score(train_y, prediction))
print "Confusion Matrix ->\n",cm
print("\nExecuted Successfully")
