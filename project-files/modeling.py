"""

    Ben Schedin
    Tumor Classification With Fine Needle Aspiration Data
    October 2021
    Statistical Modeling Script

"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


def main():
    # Loading preprocessed data
    full = pd.read_csv("../data/processed/full.csv", header=None)
    scaled = pd.read_csv("../data/processed/full-scaled.csv", header=None)
    pca = pd.read_csv("../data/processed/full-scaled.csv", header=None)
    pca7 = pd.read_csv("../data/processed/pca7.csv", header=None)

    # Prototyping code with MLP
    # data prep
    train_data, test_data, train_labels, test_labels = prep_data(scaled, 0.25)

    # model implementation
    max_epochs = 1000
    mlp = MLPClassifier(max_iter=max_epochs, random_state=123, verbose=1)
    mlp.fit(train_data, train_labels)
    test_accuracy = round(mlp.score(test_data, test_labels) * 100, 2)

    print(f"final accuracy: {test_accuracy}%")


def prep_data(data, test_percent):
    """ Takes a preprocessed data file and creates train and test sets for sk-learn """

    # Splitting labels from the data
    labels = data[0]
    data = data.iloc[:, 1:len(data.columns)]

    # Splitting data into train and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=test_percent, random_state=123)

    # Returning prepared data
    return train_data, test_data, train_labels, test_labels


if __name__ == "__main__":
    main()
