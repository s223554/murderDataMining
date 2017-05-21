import numpy as np
import numpy.random as rnd
import os


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.interactive(True)

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "output_figs"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


import pandas as pd

MURDER_PATH = "datasets/murder"

def load_data(path=MURDER_PATH):
    csv_path = os.path.join(path, "database.csv")
    return pd.read_csv(csv_path)

murder_data = load_data()

murder_data['Victim Age'].hist(bins = 300,figsize=(11,8))
plt.axis([0,100,0,100000])

# split dataset into test.

from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(murder_data, test_size=0.2, random_state=42)


corr_matrix = murder_data.corr()
corr_matrix["Victim Age"].sort_values(ascending=False)

from sklearn.preprocessing import LabelBinarizer,LabelEncoder

encoder = LabelBinarizer()
encoder.fit_transform(murder_data['Weapon'])

index1 = murder_data['Perpetrator Race']=='White'
index2 = murder_data['Perpetrator Race']=='Black'
prepared_data = murder_data[index1].append(murder_data[index2])

train_set, test_set = train_test_split(prepared_data, test_size=0.2, random_state=42)
label_enconder = LabelEncoder()
y_train = label_enconder.fit_transform(train_set['Perpetrator Race'])
y_test = label_enconder.fit_transform(test_set['Perpetrator Race'])
# property choosing



from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import Imputer, StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

#cat_attribs = ['State','Crime Type','Victim Ethnicity','Perpetrator Sex','Relationship','Weapon'\
#               ,'Victim Sex','Victim Race']
cat_attribs = ['Victim Race']
num_attribs = ["Incident",'Victim Age','Victim Count','Perpetrator Count']


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelBinarizer().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelBinarizer().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
 #       ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('encoder', LabelBinarizer()),
    ])

preparation_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

murder_train = preparation_pipeline.fit_transform(train_set)
murder_test = preparation_pipeline.fit_transform(test_set)

from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(murder_train, y_train)

sgd_clf.predict(murder_test)

from sklearn.model_selection import cross_val_score,cross_val_predict
cross_val_score(sgd_clf, murder_train, y_train, cv=3, scoring="accuracy")

from sklearn.metrics import confusion_matrix,precision_score, recall_score, precision_recall_curve
y_train_pred = cross_val_predict(sgd_clf, murder_train, y_train, cv=3)
confusion_matrix(y_train, y_train_pred)

precision_score(y_train, y_train_pred)
recall_score(y_train, y_train_pred)

y_scores = cross_val_predict(sgd_clf, murder_train, y_train, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train, y_scores)

#plot precision recall curve

def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
    plt.xlabel("Threshold", fontsize=16)
    plt.legend(loc="center left", fontsize=16)
    plt.ylim([0, 1])

plt.figure(figsize=(8, 4))
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.xlim([-3, 3])



def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])

plt.figure(figsize=(8, 6))
plot_precision_vs_recall(precisions, recalls)

# plot ROC curve

from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_train, y_scores)

def plot_roc_curve(fpr, tpr, **options):
    plt.plot(fpr, tpr, linewidth=2, **options)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)

plt.figure(figsize=(8, 6))
plot_roc_curve(fpr, tpr)
