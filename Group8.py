import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier

# Load the data into a data frame.
data_frame = pd.read_csv("dataset08.csv")
print(data_frame)

# Clean the data if needed.
data_frame = data_frame.fillna(0)  # NaN->0
print(data_frame)

# Divide the data into training and testing.
d = {'Pending': 0, 'Complete': 1}
data_frame['Or_pay_status'] = data_frame['Or_pay_status'].map(d)

train_data, test_data = train_test_split(data_frame, test_size=0.2, random_state=25)
# print(train_data)
# print(test_data)

# Input Variables
features = ['Pr_price', 'Offer_price', 'Discount_price', 'Or_total_price']

X = train_data[features]
y = train_data['Or_pay_status']  # target variable

X_test = test_data[features]
y_test = test_data['Or_pay_status']

# #knn
knn = KNeighborsClassifier(n_neighbors=4)
knn.fit(X, y)
pred_knn = knn.predict(X_test)
cm_knn = confusion_matrix(y_test, pred_knn)
print("KNN Confusion Matrix : \n", cm_knn)
print("KNN Accuracy : ", metrics.accuracy_score(y_test, pred_knn) * 100)

# ROC curve for Decision Tree
false_positive_rate_tree, true_positive_rate_tree, threshold_tree = roc_curve(y_test, pred_knn)
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Knn')
plt.plot(false_positive_rate_tree, true_positive_rate_tree)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# # #Decision Tree
dtree = DecisionTreeClassifier()
dtree.fit(X, y)
pred_dtree = dtree.predict(X_test)
# print(pred_dtree)
cm_dtree = confusion_matrix(y_test, pred_dtree)
print("Decision Tree Confusion Matrix : \n", cm_dtree)
print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, pred_dtree) * 100)
tree.plot_tree(dtree.fit(X, y))
# # data=tree.export_graphviz(dtree,out_file="tree.dot",feature_names=features,filled=True)
# #
# #ROC curve for Decision Tree
false_positive_rate_tree, true_positive_rate_tree, threshold_tree = roc_curve(y_test, pred_dtree)
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - DecisionTree')
plt.plot(false_positive_rate_tree, true_positive_rate_tree)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# # #Naive Bayes
nbayes = GaussianNB()
nbayes.fit(X, y)
pred_nbayes = nbayes.predict(X_test)
cm_nbayes = confusion_matrix(y_test, pred_nbayes)
print("Naive Bayes Confusion Matrix :\n", cm_nbayes)
print("Naive Bayes Accuracy:", metrics.accuracy_score(y_test, pred_nbayes) * 100)

# #ROC curve for Naive Bayes
false_positive_rate_tree, true_positive_rate_tree, threshold_tree = roc_curve(y_test, pred_nbayes)
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Naive Bayes')
plt.plot(false_positive_rate_tree, true_positive_rate_tree)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# #Logistic Regression
logistic = LogisticRegression()
logistic.fit(X, y)
pred_logistic = logistic.predict(X_test)
cm_logistic = confusion_matrix(y_test, pred_logistic)
print(cm_logistic)
print("logistic Accuracy:", metrics.accuracy_score(y_test, pred_logistic) * 100)
print(r2_score(y_test, pred_logistic))

# ROC curve for Logistic Regression
false_positive_rate_tree, true_positive_rate_tree, threshold_tree = roc_curve(y_test, pred_logistic)
plt.subplots(1, figsize=(10, 10))
plt.title('Receiver Operating Characteristic - Logistic Regression')
plt.plot(false_positive_rate_tree, true_positive_rate_tree)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0], c=".7"), plt.plot([1, 1], c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# k fold cross validation for enhancing(increasing) accuracy
X1 = data_frame[features]
y1 = data_frame['Or_pay_status']

knnk = KNeighborsClassifier()
scoresknn = cross_val_score(knnk, X1, y1, cv=10, scoring="accuracy")
# print(scoresknn)
print("Score Knn", scoresknn.mean() * 100)

dtreek = DecisionTreeClassifier()
scoresdtree = cross_val_score(dtreek, X1, y1, cv=10, scoring="accuracy")
# print(scoresdtree)
print("Score Dtree", scoresdtree.mean() * 100)

nbk = GaussianNB()
scoresnbk = cross_val_score(nbk, X1, y1, cv=10, scoring="accuracy")
# print(scoresnbk)
print("Score Naive Bayes", scoresnbk.mean() * 100)

logistick = LogisticRegression()
scoreslogistick = cross_val_score(logistick, X1, y1, cv=10, scoring="accuracy")
# print(scoreslogistick)
print("Score Logistic", scoreslogistick.mean() * 100)

# k-means clustering
features1 = ['Ca_state', 'Discount_price']
d1 = {'Gujarat': 1, 'Maharashtra': 2, 'Rajasthan': 3}
data_frame['Ca_state'] = data_frame['Ca_state'].map(d1)

# print(data_frame['Feedback_comment'])
x = data_frame[features1]
cluster = KMeans(n_clusters=3)
# Train the model
model = cluster.fit(x)
# print(model)
print("Clusters ", model.labels_)
centroids = cluster.cluster_centers_
print(centroids)
plt.scatter(data_frame['Ca_state'], data_frame['Discount_price'], c=cluster.labels_.astype(float), s=50, alpha=0.5)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', s=50)
plt.show()
print(data_frame['Ca_city'].unique())
