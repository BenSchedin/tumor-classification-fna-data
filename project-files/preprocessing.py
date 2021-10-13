"""

    Ben Schedin
    Tumor Classification With Fine Needle Aspiration Data
    October 2021
    Data Preprocessing Script

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing


def main():
    # ---------- IMPORT AND INSPECTION ---------- #
    # Importing raw data
    raw = pd.read_csv("../data/raw/wdbc.data.csv", header=None)

    # Inspecting imported data
    print(raw.head())
    print(raw.shape)

    # Checking for missing values
    print(raw.isnull().sum())

    # Checking for sample bias
    print(raw[1].value_counts())

    # ---------- RESHAPING ---------- #
    # Removing sample ID column and resetting column indices
    preprocessed = raw
    del preprocessed[0]
    preprocessed.columns = range(preprocessed.shape[1])
    print(preprocessed.head())

    # ---------- REMOVING SAMPLE BIAS ---------- #
    # Splitting data into two dataframes, one with benign and the other with malignant samples
    benign = preprocessed[preprocessed[0] == "B"]
    malignant = preprocessed[preprocessed[0] == "M"]

    # Calculating number of samples to remove and re-balancing
    remove_count = len(benign) - len(malignant)
    print("Removing {} benign samples.".format(remove_count))
    indices = np.random.choice(benign.index, remove_count, replace=False)
    benign = benign.drop(indices)

    # Combining data back together and shuffling to randomly intermix benign and malignant samples
    preprocessed = benign.append(malignant, ignore_index=True)
    preprocessed = preprocessed.sample(frac=1).reset_index(drop=True)
    preprocessed.columns = range(preprocessed.shape[1])
    print(preprocessed.head())
    print(preprocessed.shape)

    # ---------- SCALING THE DATA ---------- #
    # Splitting off real-valued features
    numeric = preprocessed.iloc[:, 1:31]

    # Scaling selected features
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric)
    scaled = scaler.transform(numeric)
    scaled = pd.DataFrame(scaled)
    print(scaled.head())

    # Recombining scaled features with labels and resetting indices
    preprocessed = pd.concat([preprocessed[0], scaled], axis=1)
    preprocessed.columns = range(preprocessed.shape[1])

    # Examining scaled data
    print(preprocessed.head())
    print(preprocessed.shape)

    # ---------- EXPORTING PREPROCESSED DATA ---------- #
    preprocessed.to_csv("../data/processed/full.csv", header=False, index=False)


if __name__ == "__main__":
    main()
