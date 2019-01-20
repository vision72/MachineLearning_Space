from sklearn.tree import DecisionTreeRegressor
import pandas as pd
interview_dataset = pd.read_csv('C:\Users\Admin\Desktop\~Vision\Vision\Machine Learning\~vision_datasets\dataset1.csv')
prediction_target = interview_dataset.Pass
sel_data = ['Age','Skills','Marks']
training_data = interview_dataset[sel_data]
tree_regressor_model = DecisionTreeRegressor()
tree_regressor_model.fit(training_data, prediction_target)
print(tree_regressor_model.predict(training_data))
##test_data = {'Age':19,'Skills':5,'Marks':78}
##df_test_data = pd.DataFrame(test_data)
##print(tree_regressor_model.predict(df_test_data))
####print(tree_regressor_model.predict(test_data))
####print(type(training_data))
