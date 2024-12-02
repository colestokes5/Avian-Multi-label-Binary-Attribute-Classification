"""
File: compile_cub_attributes.py
Author: Cole Stokes
Date: 2024-11-15
Last Modified: 2024-11-19
Description: Compiles image files paths and attribute information for birds from the CUB-200-2011 Dataset.
Attributes are compiled into a column of 1-d arrays of 312 attributes labeled 0 for not present and 1 for present.
"""

import pandas as pd


def get_attributes_binary(attributes_df):
    """
    Reads in the attribute DataFrame and convert the data for each image_id into binary list of length 312.
    :param df: DataFrame
    :return: List[int]
    """
    column = []
    for image_id in range(1, max(attributes_df["image_id"]) + 1):
        attribute_list = []
        image_attributes = attributes_df[attributes_df["image_id"] == image_id]
        for _, row in image_attributes.iterrows():
            attribute_list.append(int(row["is_present"]))

        column.append(attribute_list)

    return column


if __name__ == "__main__":
    # Loads the necessary data from CUB_200_2011.
    print("Creating CSV...")
    CUB_200_2011_df = pd.read_csv("CUB_200_2011/images.txt", sep=" ", header=None, names=["image_id", "path"])
    attributes_df = pd.read_csv("CUB_200_2011/attributes/image_attribute_labels.txt", sep=r"\s+", header=None, names=["image_id", "attribute_id", "is_present"], usecols=[0, 1, 2])

    # Complete the paths and fix image_id.
    CUB_200_2011_df["image_id"] = pd.to_numeric(CUB_200_2011_df["image_id"])
    CUB_200_2011_df["path"] = "CUB_200_2011/images/" + CUB_200_2011_df["path"]

    # Adds the attribute columns to the DataFrame.
    CUB_200_2011_df["attributes"] = get_attributes_binary(attributes_df)

    # Saves the DataFrame to a CSV.
    csv_name = "cub_200_2011.csv"
    CUB_200_2011_df.to_csv(csv_name, index=False)
    print(f"Saved {csv_name} to current directory.")
