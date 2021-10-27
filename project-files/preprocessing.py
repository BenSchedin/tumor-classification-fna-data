"""

    Ben Schedin
    Tumor Classification With Fine Needle Aspiration Data
    October 2021
    Data Preprocessing Script

"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


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

    # ---------- RESHAPING AND REMOVING SAMPLE BIAS ---------- #
    # Removing sample ID column and resetting column indices
    cleaned = raw
    del cleaned[0]
    cleaned.columns = range(cleaned.shape[1])
    print(cleaned.head())

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

    # Exporting cleaned data
    cleaned.to_csv("../data/processed/full.csv", header=False, index=False)

    # ---------- SCALING THE DATA ---------- #
    # Splitting off real-valued features
    numeric = cleaned.iloc[:, 1:31]

    # Scaling selected features
    scaler = preprocessing.StandardScaler()
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

    # Exporting scaled data
    scaled.to_csv("../data/processed/full-scaled.csv", header=False, index=False)

    # ---------- APPLYING PCA TO THE DATA ---------- #
    # Splitting off real-valued features
    numeric = scaled.iloc[:, 1:31]

    # Applying PCA to the selected features
    pca_solver = PCA(n_components=30, svd_solver="auto")
    components = pca_solver.fit_transform(numeric)
    pca = pd.DataFrame(components)

    # Recombining components with labels and resetting indices
    pca = pd.concat([scaled[0], pca], axis=1)
    pca.columns = range(pca.shape[1])

    # Examining PCA data
    print(pca.head())
    print(pca.shape)

    # Plotting component variance
    component_numbers = np.arange(pca_solver.n_components_) + 1
    component_variance = pca_solver.explained_variance_ratio_
    plt.plot(component_numbers, component_variance, "ro-", linewidth=2)
    plt.title("Scree Plot")
    plt.xlabel("Component")
    plt.ylabel("Variance Explained")
    plt.savefig("../figures/pca-variance.png", format="png")

    # Exporting PCA data
    pca.to_csv("../data/processed/full-pca.csv", header=False, index=False)

    # Selecting top 7 components with labels (elbow method)
    pca7 = pca.iloc[:, 0:8]

    # Exporting top components
    pca7.to_csv("../data/processed/pca7.csv", header=False, index=False)


if __name__ == "__main__":
    main()
