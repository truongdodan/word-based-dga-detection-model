import pandas as pd
from sklearn.utils import shuffle

dataset = pd.read_csv(r'D:\HOCTAP\Machine_Learning\detect_word-based_dga\dataset\final_dataset.csv')
""" dataset = shuffle(dataset)  # Shuffle rows
dataset.reset_index(inplace=True, drop=True) """

# Spliting data into traing and testing sets   
from sklearn.model_selection import train_test_split 
X = dataset.drop('isDGA', axis=1)
y = dataset['isDGA']
y = y.astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=69)

# DECISION TREE implementation - accu: 0.9661225079456804
""" # Start the model 
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()

# Training the Model
dtc.fit(X_train, y_train) 

# Testing model
y_pred = dtc.predict(X_test) 
print(r"Decision Tree's score: ") """

# NAIVE BAYES implementation - accu: 0.7701892516613695
""" from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()

gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test) 
print(r"Naive Bayes's score: ") """

# SVM implementation
""" from sklearn.svm import SVC
model = SVC()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(r"SVM's score: ") """

# RANDOM FOREST implementation _ accu: 0.972081768275065
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()

rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
print(r"Random Forest's score: ")

# Access model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

""" # Export model
import joblib
joblib.dump(rf, r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\code\dga_detecting_model.pkl") """



""" # dataset.to_csv(r'D:\HOCTAP\Machine_Learning\detect_word-based_dga\dataset\final_dataset.csv', index=None, header=True) # Export dataset   
with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.expand_frame_repr", False):
    print(dataset.head(20)) """