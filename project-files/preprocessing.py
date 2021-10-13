"""

    Ben Schedin
    Tumor Classification With Fine Needle Aspiration Data
    October 2021
    Data Preprocessing Script

"""

import pandas as pd


def main():
    # Importing raw data
    raw = pd.read_csv("../data/raw/wdbc.data.csv", header=None)
    print(raw.head())

    # Checking for missing values


if __name__ == "__main__":
    main()
