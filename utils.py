import os
import pandas as pd

def load_dataset(dataset):

    if dataset in ["movielens-1m", "amazon", "yahoo-r3"]:
        base_folder = "real_data/{}/".format(dataset)
    else:
        base_folder = "data_generation_files/{}/".format(dataset)
        for folder in os.listdir(base_folder):
            if not folder.startswith('.'):
                base_folder = os.path.join(base_folder, folder)
                break
        for folder in os.listdir(base_folder):
            if not folder.startswith('.'):
                base_folder = os.path.join(base_folder, folder)
                break

    if dataset == "movielens-1m":
        data = pd.read_csv(os.path.join(base_folder, "ratings.dat"), sep="::", engine="python", header=None)
    elif dataset == "amazon":
        dataset_path = os.path.join(base_folder, "amazon.csv")
        data = pd.read_csv(dataset_path)
        data = data[["reviewerID", "itemID"]].rename(columns={"reviewerID": 0, "itemID": 1})
    elif "yahoo-r3" == dataset:
        data = pd.read_csv(os.path.join(base_folder, "train.txt"), sep="\t", engine="python", header=None)
    else:
        data = pd.read_csv(os.path.join(base_folder, "{}.tsv".format(dataset)), sep="\t", engine="python")
        data[0] = data['User']
        data[1] = data['Item']
        data = data[[0, 1]]

    print("Data loaded.")
    return data


def print_statistics(data):

    n_users = len(data[0].unique())
    n_items = len(data[1].unique())
    print("N interactions: ", len(data))
    print("N users: ", n_users)
    print("N items: ", n_items)
    print("Density: ", len(data)/(n_users*n_items))

    return len(data)/(n_users*n_items)
