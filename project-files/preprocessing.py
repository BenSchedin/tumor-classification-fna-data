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
    cleaned = raw
    del cleaned[0]
    cleaned.columns = range(cleaned.shape[1])
    print(cleaned.head())

    # ---------- REMOVING SAMPLE BIAS ---------- #
    # Splitting data into two dataframes, one with benign and the other with malignant samples
    benign = cleaned[cleaned[0] == "B"]
    malignant = cleaned[cleaned[0] == "M"]

    # Calculating number of samples to remove and re-balancing
    remove_count = len(benign) - len(malignant)
    print("Removing {} benign samples.".format(remove_count))
    indices = np.random.choice(benign.index, remove_count, replace=False)
    benign = benign.drop(indices)

    # Combining data back together and shuffling to randomly intermix benign and malignant samples
    cleaned = benign.append(malignant, ignore_index=True)
    cleaned = cleaned.sample(frac=1).reset_index(drop=True)
    cleaned.columns = range(cleaned.shape[1])
    print(cleaned.head())
    print(cleaned.shape)

    # ---------- EXPORTING CLEANED DATA ---------- #
    cleaned.to_csv("../data/processed/full.csv", header=False, index=False)

    # ---------- SCALING THE DATA ---------- #
    # Splitting off real-valued features
    numeric = cleaned.iloc[:, 1:31]

    # Scaling selected features
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(numeric)
    scaled = scaler.transform(numeric)
    scaled = pd.DataFrame(scaled)
    print(scaled.head())

    # Recombining scaled features with labels and resetting indices
    scaled = pd.concat([cleaned[0], scaled], axis=1)
    scaled.columns = range(scaled.shape[1])

    # Examining scaled data
    print(scaled.head())
    print(scaled.shape)

    # ---------- EXPORTING SCALED DATA ---------- #
    scaled.to_csv("../data/processed/full-scaled.csv", header=False, index=False)


if __name__ == "__main__":
    main()
