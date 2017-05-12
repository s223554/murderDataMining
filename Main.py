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
test_set.head()
