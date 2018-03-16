from predict import digitize_columns
from predict import standardize_columns
from predict import encode_columns_nn
from predict import feature_scale_columns
from predict import load_model_data
from predict import execute_classifier
from predict import prepare_model_data

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn import tree
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import os.path


#model_data =  prepare_model_data(rewrite_files=True)

model_data = load_model_data()
# KNN
classifier = KNeighborsClassifier(n_neighbors=3)
execute_classifier("K - N N", classifier, model_data['train_x_nb'], model_data['train_y'], model_data['test_x_nb'], model_data['test_y'])

# naive bayes
classifier = MultinomialNB(alpha=1)
execute_classifier("Naive Bayes", classifier, model_data['train_x_nb'], model_data['train_y'], model_data['test_x_nb'], model_data['test_y'])


# logistic regression
classifier = LogisticRegression()
execute_classifier("Log Reg", classifier, model_data['train_x_nn'], model_data['train_y'], model_data['test_x_nn'], model_data['test_y'])


# neural net
classifier = MLPClassifier(activation='relu', solver='adam', hidden_layer_sizes=(128,), max_iter=300)
execute_classifier("N Net", classifier, model_data['train_x_nn'], model_data['train_y'], model_data['test_x_nn'], model_data['test_y'])

#Decision Tree
#classifier = tree.DecisionTreeClassifier()
#execute_classifier("Dec tree", classifier, model_data['train_x_nn'], model_data['train_y'], model_data['test_x_nn'], model_data['test_y'])

#classifier = GradientBoostingClassifier(n_estimators=300)
#execute_classifier("Grad Boost", classifier, model_data['train_x_nn'], model_data['train_y'], model_data['test_x_nn'], model_data['test_y'])
