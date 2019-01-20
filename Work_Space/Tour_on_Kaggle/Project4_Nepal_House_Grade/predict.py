import pandas as pd
train_data_x  = pd.read_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project4_datasets\\Nepal_train_values.csv')
test_data_x   = pd.read_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project4_datasets\\nepaltestsetvalues.csv')
train_data_y  = pd.read_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project4_datasets\\Nepal_dam_grade.csv')
sel_vals = ['count_floors_pre_eq', 'age','area','height']
train_X  = train_data_x[sel_vals]
test_X   = test_data_x[sel_vals]
train_y  = train_data_y.damage_grade
building_id = train_data_y.building_id

from sklearn.tree import DecisionTreeRegressor
clf = DecisionTreeRegressor()
clf.fit(train_X, train_y)
prediction = clf.predict(test_X)

prediction = prediction.astype(int)

my_submission = pd.DataFrame({'Id': building_id, 'SalePrice': prediction})
my_submission.to_csv('C:\\Users\\Admin\\Desktop\\~Vision\\Vision\\Machine Learning\\~Datasets\\~vision_datasets\\Project4_datasets\\Validated_dataset\\submission.csv', index=False)
