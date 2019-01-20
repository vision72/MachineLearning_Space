from sklearn import tree, ensemble, metrics, svm
X = [[65, 112], [71, 136], [69, 153], [68, 142], [67, 144]]
y = ['female', 'male', 'male', 'female', 'female']
clf = tree.DecisionTreeClassifier()
clf2 = ensemble.RandomForestClassifier()
clf3 = svm.SVC()
acc = metrics.accuracy_score
clf.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
test_X = [[66, 112], [70, 136], [68, 153], [69, 142], [72, 144]]
pred1 = clf.predict(test_X)
pred2 = clf2.predict(test_X)
pred3 = clf3.predict(test_X)
print acc(y, pred1)*100
print acc(y, pred2)*100
print acc(y, pred3)*100