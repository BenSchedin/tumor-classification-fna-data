"""

    Ben Schedin
    Tumor Classification With Fine Needle Aspiration Data
    October 2021
    Data Preprocessing Script

"""

import pandas as pd
import numpy as np


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
    del raw[0]
    raw.columns = range(raw.shape[1])
    print(raw.head())

    # ---------- REMOVING SAMPLE BIAS ---------- #
    # Splitting data into two dataframes, one with benign and the other with malignant samples
    benign = raw[raw[0] == "B"]
    malignant = raw[raw[0] == "M"]

    # Calculating number of samples to remove and re-balancing
    remove_count = len(benign) - len(malignant)
    print("Removing {} benign samples.".format(remove_count))
    indices = np.random.choice(benign.index, remove_count, replace=False)
    benign = benign.drop(indices)

    # Combining data back together and shuffling to randomly intermix benign and malignant samples




if __name__ == "__main__":
    main()
