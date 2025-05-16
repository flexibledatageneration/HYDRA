import torch
from torch import optim
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
import math
import os
import zipfile
import requests
import argparse

from estimation_model import Network_loss, Q_phi_Network, StretchedExponential, PowerLaw, Lognormal, Exponential


def download_movielens_1m(output_dir="real_data/movielens_1m"):
    # URL for the MovieLens 1M dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    dataset_zip_path = os.path.join(output_dir, "ml-1m.zip")

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Download the dataset
    print("Downloading MovieLens 1M dataset...")
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(dataset_zip_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Dataset downloaded successfully and saved to {dataset_zip_path}")
    else:
        raise Exception(f"Failed to download dataset. HTTP Status Code: {response.status_code}")

    # Extract the dataset
    print("Extracting the dataset...")
    with zipfile.ZipFile(dataset_zip_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    print(f"Dataset extracted successfully to {output_dir}")

    # Clean up the zip file
    os.remove(dataset_zip_path)
    print(f"Cleaned up zip file: {dataset_zip_path}")


def load_dataset(dataset):
    base_folder = "real_data/{}/".format(dataset)

    if not os.path.exists("real_data"):
        os.makedirs("real_data")

    if dataset == "movielens-1m":
        if not os.path.exists("real_data/movielens_1m"):
            download_movielens_1m()
        data = pd.read_csv(os.path.join("movielens_1m/ml-1m", "ratings.dat"), sep="::", engine="python", header=None)
    elif dataset == "amazon":
        dataset_path = os.path.join(base_folder, "amazon.csv")
        data = pd.read_csv(dataset_path)
        data = data[["reviewerID", "itemID"]].rename(columns={"reviewerID": 0, "itemID": 1})
    elif "yahoo-r3" == dataset:
        data = pd.read_csv(os.path.join(base_folder, "train.txt"), sep="\t", engine="python", header=None)

    print("Data loaded.")
    return data

def compute_degree_distributions(data):

    bottom_grouped_df = data.groupby(0)  # users
    top_grouped_df = data.groupby(1)  # items

    user_items = []
    item_users = []

    for _, group in bottom_grouped_df:
        user_items.append(len(group))

    for _, group in top_grouped_df:
        item_users.append(len(group))

    bottom_x, bottom_distribution = np.unique(user_items, return_counts=True)
    top_x, top_distribution = np.unique(item_users, return_counts=True)

    return top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users


def negative_sampling(num_users, num_items, interactions, num_negative_interactions=10):
    neg_interactions = []
    items_total = np.arange(num_items)

    for u in range(num_users):
        items_u = interactions[interactions[:, 0] == u][:, 1]
        num_interactions_u = len(items_u)
        items_available = list(set(items_total) - set(items_u))
        size_to_take = num_interactions_u * num_negative_interactions
        if len(items_available) < size_to_take:
            size_to_take = len(items_available)
        neg_interactions_u = np.random.choice(items_available, size_to_take, replace=False)
        neg_interactions.extend(
            list(zip(np.repeat(u, size_to_take), neg_interactions_u)))

    negative_items = np.unique(np.array(neg_interactions)[:, 1])
    if len(negative_items) < num_items:
        users_total = np.arange(num_users)
        items_available = list(set(items_total) - set(negative_items))
        for i in items_available:
            neg_interactions_i = np.random.choice(users_total, 1, replace=False)
            neg_interactions.append([neg_interactions_i[0], i])

    return neg_interactions

# Initialize parser
parser = argparse.ArgumentParser()

# Adding optional argument
parser.add_argument("-d", "--dataset", help = "Specify dataset name")
parser.add_argument("-o", "--objective", help = "Specify what to optimize between 'longtails' and 'preferences'")
parser.add_argument("-ud", "--user_dist", help = "Specify the user distribution (PowerLaw, Exponential, StretchedExponential, Lognormal)")
parser.add_argument("-id", "--item_dist", help = "Specify the item distribution (PowerLaw, Exponential, StretchedExponential, Lognormal)")

# Read arguments from command line
args = parser.parse_args()

dataset = args.dataset
batch_size = 2**11

num_epochs = 10000
learning_rate = 0.001
num_epochs_early_stopping = 40
num_negative_interactions_per_user = 1
objective = args.objective # longtails, preferences

device_string = "cpu"
device = torch.device(device_string)

data = load_dataset(dataset)

user_count = 0
users_dict = {}

item_count = 0
items_dict = {}

interactions = data[[0, 1]].to_numpy()
data = []
for u, i in interactions:

    if u not in users_dict.keys():
        users_dict[u] = user_count
        user_count += 1

    if i not in items_dict.keys():
        items_dict[i] = item_count
        item_count += 1

    data.append((users_dict[u], items_dict[i]))

data = pd.DataFrame(data, columns=[0, 1])

interactions = data[[0, 1]].to_numpy()

_, _, _, _, users_degree, items_degree = compute_degree_distributions(data)

num_users = interactions[:, 0].max() + 1
num_items = interactions[:, 1].max() + 1

density = torch.Tensor([len(interactions) / (num_users * num_items)])

interactions_neg = negative_sampling(num_users, num_items, interactions, num_negative_interactions=num_negative_interactions_per_user)
interactions_neg_df = pd.DataFrame(interactions_neg, columns=[0, 1])

data[2] = np.ones(len(data))
interactions_neg_df[2] = np.zeros(len(interactions_neg))

print(interactions_neg_df.shape, data.shape)

if objective == "longtails":
    interactions_pos_neg = torch.LongTensor(np.array(data))
else:
    interactions_pos_neg = torch.LongTensor(pd.concat((data, interactions_neg_df)).to_numpy())

train_set, val_set = torch.utils.data.random_split(interactions_pos_neg, [0.8, 0.2])
train_set = interactions_pos_neg[train_set.indices]

# Check if all users appear in train_set
if len(np.unique(train_set[:, 0])) < num_users:
    users_already_in_set = set(np.unique(train_set[:, 0]))
    missing_users = set(np.arange(0, num_users)) - users_already_in_set
    for u in missing_users:
        item = interactions[interactions[:, 0] == u][0][1]
        train_set = np.append(train_set, values=[[u, item, 1]], axis=0)

# Check if all items appear in train_set
if len(np.unique(train_set[:, 1])) < num_items:
    items_already_in_set = set(np.unique(train_set[:, 1]))
    missing_items = set(np.arange(0, num_items)) - items_already_in_set
    for i in missing_items:
        user = interactions[interactions[:, 1] == i][0][0]
        train_set = np.append(train_set, values=[[user, i, 0]], axis=0)

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

hidden_dim_users_items = int(math.sqrt(num_items)) if num_items > num_users else int(math.sqrt(num_users))

users_degree_relative = np.array(users_degree) / num_items
items_degree_relative = np.array(items_degree) / num_users

users_degree = torch.Tensor(users_degree)
items_degree = torch.Tensor(items_degree)

users_degree_relative = torch.Tensor(users_degree_relative).to(device)
items_degree_relative = torch.Tensor(items_degree_relative).to(device)

xmin_users = users_degree.min().to(device)
xmin_items = items_degree.min().to(device)

xmax_users = users_degree.max()
xmax_items = items_degree.max()

print("Num Interactions", len(interactions))
print("XMIN USERS", xmin_users.item(), "XMAX USERS", xmax_users.item(), "XMIN ITEMS", xmin_items.item(), "XMAX ITEMS", xmax_items.item())
print("Num users", num_users, "Num items", num_items)

probs_user = []
if args.user_dist == "PowerLaw":
    probs_user.append(PowerLaw(xmin=xmin_users, device=device))
elif args.user_dist == "Exponential":
    probs_user.append(Exponential(xmin=xmin_users, device=device))
elif args.user_dist == "StretchedExponential":
    probs_user.append(StretchedExponential(xmin=xmin_users, device=device))
elif args.user_dist == "Lognormal":
    probs_user.append(Lognormal(xmin=xmin_users, device=device))
else:
    raise Exception("User distribution not available")

probs_item = []
if args.item_dist == "PowerLaw":
    probs_item.append(PowerLaw(xmin=xmin_items, device=device))
elif args.item_dist == "Exponential":
    probs_item.append(Exponential(xmin=xmin_items, device=device))
elif args.item_dist == "StretchedExponential":
    probs_item.append(StretchedExponential(xmin=xmin_items, device=device))
elif args.item_dist == "Lognormal":
    probs_item.append(Lognormal(xmin=xmin_items, device=device))
else:
    raise Exception("Item distribution not available")

Q_phi = Q_phi_Network(hidden_dim_users_items, num_users, num_items, probs_user, probs_item, batch_size, device).to(device)

parameters = []
parameters.extend(list(Q_phi.parameters()))

for dist in probs_user:
    parameters.extend(list(dist.parameters()))

for dist in probs_item:
    parameters.extend(list(dist.parameters()))

network_loss = Network_loss(device)
optimizer = optim.Adam(parameters, lr=learning_rate)

print("Inizialization params")
print("Users:")
for dist in Q_phi.probs_user:
    print("Dist", dist.name, "Params", dist.get_params())

print("Items:")
for dist in Q_phi.probs_item:
    print("Dist", dist.name, "Params", dist.get_params())

print("Start")

path = "{}_{}_{}_{}_{}".format(dataset, Q_phi.probs_user[0].name, Q_phi.probs_item[0].name, learning_rate, num_negative_interactions_per_user)

if not os.path.exists("estimation_files/{}".format(path)):
    os.makedirs("estimation_files/{}".format(path))

if not os.path.exists("estimation_files/{}/plot_loss".format(path)):
    os.makedirs("estimation_files/{}/plot_loss".format(path))

priors_longtail_users_train_list = []
priors_longtail_items_train_list = []
priors_dirichlet_train_list = []
preferences_train_list = []
priors_beta_train_list = []

priors_longtail_users_val_list = []
priors_longtail_items_val_list = []
priors_dirichlet_val_list = []
preferences_val_list = []
priors_beta_val_list = []

mse = torch.nn.MSELoss()

best_loss_val = None #0.0
save_best = True
patience = 0
try:
    for epoch in tqdm(range(num_epochs)):
        losses_train = []
        priors_longtail_users_train = 0
        priors_longtail_items_train = 0
        priors_dirichlet_train = 0
        preferences_train = 0
        priors_beta_train = 0
        for i, batch_interactions in enumerate(train_dataloader):
            optimizer.zero_grad()

            batch_interactions = batch_interactions
            users, items, ratings = batch_interactions[:, 0], batch_interactions[:, 1], batch_interactions[:, 2]

            Q_phi.train()

            params = Q_phi(users.to(device), items.to(device), users_degree.to(device),
                           items_degree.to(device), density.to(device))

            loss, priors_longtail_users, priors_longtail_items, priors_dirichlet, preferences, \
                priors_beta, softmax_matrix_k_h = network_loss(params, users_degree_relative[users], items_degree_relative[items],
                                           ratings.float().to(device), objective)

            loss.backward()

            # update weights
            optimizer.step()

            losses_train.append(loss.item())
            priors_longtail_users_train += priors_longtail_users.item()
            priors_longtail_items_train += priors_longtail_items.item()
            priors_dirichlet_train += priors_dirichlet.item()
            preferences_train += preferences.item()
            priors_beta_train += priors_beta.item()

        priors_longtail_users_train_list.append(priors_longtail_users_train / len(train_dataloader))
        priors_longtail_items_train_list.append(priors_longtail_items_train / len(train_dataloader))
        priors_dirichlet_train_list.append(priors_dirichlet_train / len(train_dataloader))
        preferences_train_list.append(preferences_train / len(train_dataloader))
        priors_beta_train_list.append(priors_beta_train / len(train_dataloader))

        # VALIDATION
        with torch.no_grad():
            losses_val = []
            priors_longtail_users_val = 0
            priors_longtail_items_val = 0
            priors_dirichlet_val = 0
            preferences_val = 0
            priors_beta_val = 0
            for i, batch_interactions in enumerate(val_dataloader):
                optimizer.zero_grad()

                batch_interactions = batch_interactions.to(device)
                users, items, ratings = batch_interactions[:, 0], batch_interactions[:, 1], batch_interactions[:, 2]

                params = Q_phi(users.to(device), items.to(device), users_degree.to(device),
                               items_degree.to(device), density.to(device))

                loss, priors_longtail_users, priors_longtail_items, priors_dirichlet, preferences, \
                    priors_beta, _ = network_loss(params, users_degree_relative[users], items_degree_relative[items],
                                               ratings.float(), objective)

                losses_val.append(loss.item())
                priors_longtail_users_val += priors_longtail_users.item()
                priors_longtail_items_val += priors_longtail_items.item()
                priors_dirichlet_val += priors_dirichlet.item()
                preferences_val += preferences.item()
                priors_beta_val += priors_beta.item()

        priors_longtail_users_val_list.append(priors_longtail_users_val / len(val_dataloader))
        priors_longtail_items_val_list.append(priors_longtail_items_val / len(val_dataloader))
        priors_dirichlet_val_list.append(priors_dirichlet_val / len(val_dataloader))
        preferences_val_list.append(preferences_val / len(val_dataloader))
        priors_beta_val_list.append(priors_beta_val / len(val_dataloader))

        if objective == "longtails":
            print(f'Epoch: {epoch}')
            print("Probs distributions:")
            for k, dist_k in enumerate(Q_phi.probs_user):
                for h, dist_h in enumerate(Q_phi.probs_item):
                    print("Dist User {}, Dist Item {}, prob {}".format(dist_k.name, dist_h.name,
                                                                       softmax_matrix_k_h[k, h]))

            print("Users:")
            for d, dist in enumerate(Q_phi.probs_user):
                print("Dist", dist.name, "Params", dist.get_params())

            print("Items:")
            for d, dist in enumerate(Q_phi.probs_item):
                print("Dist", dist.name, "Params", dist.get_params())

            print(f'Train: Longtail Users: {priors_longtail_users_train_list[-1]}, Longtail Items: {priors_longtail_items_train_list[-1]}')
            print(f'Val: Longtail Users: {priors_longtail_users_val_list[-1]}, Longtail Items: {priors_longtail_items_val_list[-1]}')

        if objective == "preferences":

            print(f'Epoch: {epoch}')
            print("Lambda: {}".format(torch.exp(Q_phi.mlp_density(density.to(device)).cpu().detach()).item()))

            print(f'Train:'
                  f' Dirichlet: {priors_dirichlet_train_list[-1]}, '
                  f'Preferences: {preferences_train_list[-1]}, '
                  f'Beta: {priors_beta_train_list[-1]}')

            print(f'Val:'
                  f' Dirichlet: {priors_dirichlet_val_list[-1]}, '
                  f'Preferences: {preferences_val_list[-1]}, '
                  f'Beta: {priors_beta_val_list[-1]}')

        if save_best and (best_loss_val is None \
            or (objective == "preferences" and best_loss_val > preferences_val_list[-1])\
            or (objective == "longtails" and best_loss_val > priors_longtail_users_val_list[-1] + priors_longtail_items_val_list[-1])):

            if objective == "preferences":
                best_loss_val = preferences_val_list[-1]
                torch.save(Q_phi.emb_users, "estimation_files/{}/rho.pt".format(path))
                torch.save(Q_phi.emb_items, "estimation_files/{}/alpha.pt".format(path))
                f = open("estimation_files/{}/best_conf_lambda.txt".format(path), "w")
                f.write("Lambda: {}".format(torch.exp(Q_phi.mlp_density(density.to(device)).cpu().detach()).item()))
                f.close()
            elif objective == "longtails":
                best_loss_val = priors_longtail_users_val_list[-1] + priors_longtail_items_val_list[-1]
                f = open("estimation_files/{}/best_conf_longtails.txt".format(path), "w")
                f.write("Probs distributions:\n")
                for k, dist_k in enumerate(Q_phi.probs_user):
                    for h, dist_h in enumerate(Q_phi.probs_item):
                        f.write("Dist User {}, Dist Item {}\n".format(dist_k.name, dist_h.name))

                f.write("Users:\n")
                for d, dist in enumerate(Q_phi.probs_user):
                    f.write("Dist {}, Params {} \n".format(dist.name, dist.get_params()))

                f.write("Items:\n")
                for d, dist in enumerate(Q_phi.probs_item):
                    f.write("Dist {}, Params {} \n".format(dist.name, dist.get_params()))
                f.close()

            patience = 0.0

        elif save_best:
            patience += 1

        if patience >= num_epochs_early_stopping:
            break

finally:
    import traceback

    traceback.print_exc()

    if save_best and objective == "longtails":
        plt.figure(figsize=(15, 8))
        plt.plot(priors_longtail_users_train_list, label="Train")
        plt.plot(priors_longtail_users_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("estimation_files/{}/plot_loss/priors_longtail_users.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(priors_longtail_users_train_list, label="Train")
        plt.plot(priors_longtail_users_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.yscale('log')
        plt.savefig("estimation_files/{}/plot_loss/priors_longtail_users_log.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(priors_longtail_items_train_list, label="Train")
        plt.plot(priors_longtail_items_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("estimation_files/{}/plot_loss/priors_longtail_items.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(priors_longtail_items_train_list, label="Train")
        plt.plot(priors_longtail_items_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.yscale('log')
        plt.savefig("estimation_files/{}/plot_loss/priors_longtail_items_log.png".format(path), bbox_inches='tight')

    elif save_best and objective == "preferences":

        plt.figure(figsize=(15, 8))
        plt.plot(priors_dirichlet_train_list, label="Train")
        plt.plot(priors_dirichlet_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("estimation_files/{}/plot_loss/priors_dirichlet.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(priors_dirichlet_train_list, label="Train")
        plt.plot(priors_dirichlet_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.yscale('log')
        plt.savefig("estimation_files/{}/plot_loss/priors_dirichlet_log.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(preferences_train_list, label="Train")
        plt.plot(preferences_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("estimation_files/{}/plot_loss/preferences.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(preferences_train_list, label="Train")
        plt.plot(preferences_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.yscale('log')
        plt.savefig("estimation_files/{}/plot_loss/preferences_log.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(priors_beta_train_list, label="Train")
        plt.plot(priors_beta_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.savefig("estimation_files/{}/plot_loss/priors_beta.png".format(path), bbox_inches='tight')

        plt.figure(figsize=(15, 8))
        plt.plot(priors_beta_train_list, label="Train")
        plt.plot(priors_beta_val_list, label="Val")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        plt.yscale('log')
        plt.savefig("estimation_files/{}/plot_loss/priors_beta_log.png".format(path), bbox_inches='tight')
