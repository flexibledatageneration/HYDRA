import os
import pandas as pd

def load_dataset(dataset):

    if "synthetic" in dataset:
        base_folder = "data_generation_files/{}/".format(dataset)
        for folder in os.listdir(base_folder):
            if not folder.startswith('.'):
                base_folder = os.path.join(base_folder, folder)
                break
        for folder in os.listdir(base_folder):
            if not folder.startswith('.'):
                base_folder = os.path.join(base_folder, folder)
                break
    else:
        base_folder = "real_data/{}/".format(dataset)

    if dataset == "movielens-1m":
        #dataset_path = os.path.join(base_folder, "movielens-1m")
        data = pd.read_csv(os.path.join(base_folder, "ratings.dat"), sep="::", engine="python", header=None)
    elif "movielens-1m_synthetic" in dataset:
        data = pd.read_csv(os.path.join(base_folder, "{}.tsv".format(dataset)), sep="\t", engine="python")
        data[0] = data['User']
        data[1] = data['Item']
        data = data[[0, 1]]
    elif dataset == "amazon":
        dataset_path = os.path.join(base_folder, "amazon.csv")
        data = pd.read_csv(dataset_path)
        data = data[["reviewerID", "itemID"]].rename(columns={"reviewerID": 0, "itemID": 1})
    elif "amazon_synthetic" in dataset:
        data = pd.read_csv(os.path.join(base_folder, "{}.tsv".format(dataset)), sep="\t", engine="python")
        data[0] = data['User']
        data[1] = data['Item']
        data = data[[0, 1]]
    elif "yahoo-r3" == dataset:
        data = pd.read_csv(os.path.join(base_folder, "train.txt"), sep="\t", engine="python", header=None)
    elif "yahoo_r3_synthetic" in dataset or "yahoo_r3_synthetic" in dataset:
        data = pd.read_csv(os.path.join(base_folder, "{}.tsv".format(dataset)), sep="\t", engine="python")
        data[0] = data['User']
        data[1] = data['Item']
        data = data[[0, 1]]
    else:
        print("Not yet implemented", dataset)
        exit(0)

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
