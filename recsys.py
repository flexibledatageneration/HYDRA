import os
import sys
import pandas as pd

from recbole.config import Config
from recbole.model.general_recommender import Random, Pop, ItemKNN, BPR, NeuMF, GCMC, NGCF, LightGCN, MultiVAE
from recbole.trainer.trainer import Trainer, RecVAETrainer
from recbole.data import create_dataset, data_preparation, save_split_dataloaders, load_split_dataloaders
from recbole.utils import init_seed, init_logger

from utils import *


path = "recbole/"
recbole_checkpoint_folder = os.path.join(path, "checkpoints")
recbole_data_folder = os.path.join(path, "data")

if not os.path.exists(recbole_checkpoint_folder):
    os.makedirs(recbole_checkpoint_folder)

if not os.path.exists(recbole_data_folder):
    os.makedirs(recbole_data_folder)


def generate_recbole_dataset(data, dataset, dataset_folder):
    users_dict = {}
    items_dict = {}

    uir_dataset = []

    users_count = 1
    items_count = 1

    data_array = data.to_numpy()

    for uir in data_array:

        user_id, item_id = uir[0], uir[1]

        if user_id not in users_dict:
            users_dict[user_id] = users_count
            users_count += 1

        if item_id not in items_dict:
            items_dict[item_id] = items_count
            items_count += 1

        uir_dataset.append(
            (users_dict[user_id], items_dict[item_id], 1))

    df = pd.DataFrame(uir_dataset, columns=['user_id:token', 'item_id:token', 'rating:float'])
    df.to_csv("{}/{}.inter".format(dataset_folder, dataset), index=False, sep='\t')


def run_procedure(dataset, models):

    metrics = ["Recall", "Hit"]
    topks = [5, 10, 20]

    results = []

    for model_name in models:
        data = load_dataset(dataset)

        if "amazon" in dataset:
            bottom_grouped_df = data.groupby(0)  # users
            good_users = []
            good_items = []

            for user, group in bottom_grouped_df:
                if len(group) > 4:
                    good_users.append(user)

            data = data[data[0].isin(good_users)]

            top_grouped_df = data.groupby(1)  # items
            for item, group in top_grouped_df:
                if len(group) > 4:
                    good_items.append(item)

            data = data[data[1].isin(good_items)]

        dataset_recbole_data_folder = os.path.join(recbole_data_folder, dataset)
        if not os.path.exists(dataset_recbole_data_folder):
            os.makedirs(dataset_recbole_data_folder)

        dataset_recbole_checkpoint_folder = os.path.join(recbole_checkpoint_folder, dataset)
        if not os.path.exists(dataset_recbole_checkpoint_folder):
            os.makedirs(dataset_recbole_checkpoint_folder)

        if not os.path.exists("recbole/{}".format(dataset)):
            os.makedirs("recbole/{}".format(dataset))

        if not os.path.exists("recbole/{}/tmp".format(dataset)):
            os.makedirs("recbole/{}/tmp".format(dataset))

        generate_recbole_dataset(data, dataset, dataset_recbole_data_folder)

        epochs = 100
        if model_name in ["Random", "Pop", "ItemKNN"]:
            epochs = 1

        if "amazon" in dataset:
            parameter_dict = {
                "eval_step": 5,  # 5,
                "topk": topks,
                "metrics": metrics,
                "valid_metric": "Recall@20",
                "load_col": {"inter": ["user_id", "item_id", "rating"]},
                "data_path": recbole_data_folder,
                "checkpoint_dir": dataset_recbole_checkpoint_folder,
                "epochs": epochs,
                "gpu_id": 0,
                "stopping_step": 3,
                "latent_dimendion": 16,
                "latent_dimension": 16,
                "mlp_hidden_size": [32],
            }
        else:
            parameter_dict = {
                "eval_step": 5,  # 5,
                "topk": topks,
                "metrics": metrics,
                "valid_metric": "Recall@10",
                "load_col": {"inter": ["user_id", "item_id", "rating"]},
                "data_path": recbole_data_folder,
                "checkpoint_dir": dataset_recbole_checkpoint_folder,
                "epochs": epochs,
                "gpu_id": 0,
                "stopping_step": 3,
            }

        config = Config(
            model=model_name,
            dataset=dataset,
            config_dict=parameter_dict)

        # init random seed
        init_seed(config["seed"], config["reproducibility"])

        # logger initialization
        init_logger(config)
        dataset_config = create_dataset(config)

        train_data, valid_data, test_data = data_preparation(config, dataset_config)

        module = __import__("recbole.model.general_recommender", fromlist=[model_name])
        class_ = getattr(module, model_name)
        model = class_(config, train_data.dataset).to(config["device"])

        trainer = Trainer(config, model)

        _, score = trainer.fit(train_data, valid_data, saved=True)

        results_local = trainer.evaluate(
                test_data,
                load_best_model=False,
                show_progress=False)
        print("\n Evaluation on test:")
        print(results_local)

        tmp_path = "recbole/{}/tmp/results_{}.tsv".format(dataset, dataset)

        if not os.path.isfile(tmp_path):
            f = open(tmp_path, "w")
            f.write("dataset\t")
            f.write("model\t")
            for i, metric in enumerate(metrics):
                for j, topk in enumerate(topks):

                    if i == len(metrics) - 1 and j == len(topks) - 1:
                        f.write("{}@{}\n".format(metric.lower(), topk))
                    else:
                        f.write("{}@{}\t".format(metric.lower(), topk))
            f.close()

        f = open(tmp_path, "a")

        f.write("{}\t".format(dataset))
        f.write("{}\t".format(model_name))
        for i, metric in enumerate(metrics):
            for j, topk in enumerate(topks):

                if i == len(metrics) - 1 and j == len(topks) - 1:
                    f.write("{}\n".format(results_local["{}@{}".format(metric.lower(), topk)]))
                else:
                    f.write("{}\t".format(results_local["{}@{}".format(metric.lower(), topk)]))

        f.close()

        for metric in metrics:
            for topk in topks:
                results.append((dataset, model_name, metric, topk, results_local["{}@{}".format(metric.lower(), topk)]))

    df = pd.DataFrame(results, columns=["Dataset", "RS", "Metric", "TopK", "Value"])

    for metric in metrics:
        df_metric = df[df['Metric'] == metric]
        for topk in topks:
            df_metric_topk = df_metric[df_metric['TopK'] == topk]
            df_metric_topk_grouped = df_metric_topk.sort_values(['Value', 'RS'], ascending=False).groupby('Dataset')
            new_df = []
            g = df_metric_topk_grouped.get_group(dataset)
            new_df.extend(g.to_numpy())

            new_df = pd.DataFrame(new_df, columns=["Dataset", "RS", "Metric", "TopK", "Value"])
            new_df["RS-Value"] = new_df['RS'] + ["-" for _ in range(new_df.shape[0])] + new_df['Value'].astype('str')
            new_df[["Dataset", "RS-Value"]].to_csv('recbole/{}/{}_{}.tsv'.format(dataset, metric, topk), sep="\t", index=False)




if __name__ == '__main__':

    if len(sys.argv) < 2:
        print("Error: provide a dataset name")
        exit(0)

    dataset = sys.argv[1]
    models = ["Pop", "ItemKNN", "BPR", "NeuMF", "MultiVAE"]

    run_procedure(dataset, models)
