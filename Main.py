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

murder_prepared = preparation_pipeline.fit_transform(prepared_data)
