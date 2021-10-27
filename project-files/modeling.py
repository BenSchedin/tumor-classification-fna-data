"""

    Ben Schedin
    Tumor Classification With Fine Needle Aspiration Data
    October 2021
    Statistical Modeling Script

"""

import pandas as pd
from sklearn.model_selection import train_test_split


def main():
    # Loading preprocessed data
    full = pd.read_csv("../data/processed/full.csv", header=None)
    scaled = pd.read_csv("../data/processed/full-scaled.csv", header=None)
    pca = pd.read_csv("../data/processed/full-scaled.csv", header=None)
    pca7 = pd.read_csv("../data/processed/pca7.csv", header=None)

    # Splitting the class labels from the data
    print(len(full.columns))


def prep_data(data, test_percent):
    """ Takes a preprocessed data file and creates train and test sets for sk-learn """

    # Splitting labels from the data
    labels = data[0]
    data = data.iloc[:, 1:len(data.columns)]

    # Splitting data into train and test sets


if __name__ == "__main__":
    main()
