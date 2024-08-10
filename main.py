import os
import numpy as np
from os.path import exists
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from joblib import Parallel, delayed
import powerlaw
import argparse
import json

def show_interactions_plot(data, saving_path, eps):
    #fig, ax = plt.subplots(figsize=(10, 10))
    #sns.reset_orig()
    params = {
        'figure.dpi': 200,
        "text.usetex": True,
        'legend.title_fontsize': 14,
        'legend.fontsize': 14,
        'axes.labelsize': 20,
        'axes.titlesize': 16,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        "font.family": "sans-serif",
        "font.sans-serif": "Computer Modern Sans Serif",
    }

    import matplotlib as mpl
    mpl.rcParams.update(params)
    plt.figure()

    data['Score'] = np.ones(data.shape[0])
    data_pivot = data[['Item', 'User', "Score"]].pivot(index="Item", columns="User", values="Score")

    plt.imshow(data_pivot, interpolation='nearest', aspect='auto')
    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if eps is not None:
        plt.title("$\epsilon = {}$".format(eps), fontsize=30)

    #plt.imshow(
    #    true_util.T,
    #    interpolation='nearest', aspect='auto',
    #    cmap='Greens')

    #plt.yticks(np.arange(0, final_num_items+1)[::-1])

    plt.xlabel('Users', fontsize=20)

    #     if ETA == 0.1:
    plt.ylabel('Items', fontsize=20)
    #plt.title("$\epsilon={}$".format(epsilon), fontsize=30)
    #plt.tight_layout()

    plt.tight_layout()

    plt.savefig(saving_path + f'intersections.pdf', dpi=200)

    plt.show()

    plt.clf()

    # plt.show()

def plot_degree_distributions(distributions, images_folder, category=None, population=None, top_lim=None, right_lim=None,
                              color=None, zeta=None, xi=None, Lambda=None):
    params = {
        'figure.dpi': 200,
        "text.usetex": True,
        'legend.title_fontsize': 20,
        'legend.fontsize': 20,
        'axes.labelsize': 25,
        'axes.titlesize': 16,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        "font.family": "sans-serif",
        "font.sans-serif": "Computer Modern Sans Serif",
    }

    import matplotlib as mpl
    mpl.rcParams.update(params)

    if distributions[0] is not None and distributions[2] is not None:
        #plt.figure()
        ax = plt.gca()

        sns.scatterplot(x=distributions[0], y=distributions[1], label="Items", s=100, linewidth=0.25)
        sns.scatterplot(x=distributions[2], y=distributions[3], label="Users", s=100, linewidth=0.25)

        if (zeta is not None) or (xi is not None) or (Lambda is not None):
            if zeta is not None:
                title = "$\zeta = {}$".format(zeta)
            elif xi is not None:
                title = "$\\" + "xi = " +f'{xi}$'
            elif Lambda is not None:
                exponent = len(str(Lambda).replace("1", ""))
                title = "$\lambda = 10^{}$".format(exponent)

            plt.title(title, fontsize=30)

        plt.xscale("log")
        plt.yscale("log")

        if top_lim is not None:
            plt.ylim(bottom=0.7, top=top_lim)
            yticks = [10 ** exp * x for exp in range(0, int(np.log10(top_lim))) for x in range(1, 10)]
            plt.yticks(yticks)

        if right_lim is not None:
            plt.xlim(left=0.7, right=right_lim)
            xticks = [10 ** exp * x for exp in range(0, int(np.log10(right_lim))) for x in range(1, 10)]
            plt.xticks(xticks)

        plt.xlabel("Degree")

        plt.tight_layout()

        fn = "degree_distributions.pdf"
        image_path = os.path.join(images_folder, fn)

        handles, labels = ax.get_legend_handles_labels()

        if "power_law_power_law_zeta" in images_folder or "power_law_power_law_lambda" in images_folder or "power_law_power_law_xi" in images_folder:
            ax.get_legend().remove()

        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.savefig(image_path)
        plt.show()

        '''
        # SAVE LEGEND
        legend_size = 10

        fig_legend, axi = plt.subplots(figsize=(1.91, .4))

        fig_legend.legend(handles, labels, ncol=2, fontsize=legend_size)
        axi.axis('off')
        fig_legend.tight_layout()
        fig_legend.savefig("legend.pdf", dpi=200)
        '''

        #plt.clf()

    if distributions[0] is not None:

        label = "Items"
        fn = "degree_distributions_top"

        if category is not None:
            label += "_{}".format(category)
            fn += "_{}".format(category)

        fn += ".pdf"

        plt.figure()


        if color is None:
            sns.scatterplot(x=distributions[0], y=distributions[1], label=label, legend=None, s=100, linewidth=0.25)
        else:
            hue_tmp = np.ones(len(distributions[0]))
            sns.scatterplot(x=distributions[0], y=distributions[1], hue=hue_tmp, s=100, linewidth=0.25, palette=[color])
            plt.legend(labels=['$I_{}$'.format(category + 1)])


        plt.xscale("log")
        plt.yscale("log")

        '''
        plt.ylim(bottom=0.7, top=10 ** 3)
        plt.xlim(left=0.7, right=10 ** 4)

        plt.xticks([10 ** 0, 2, 3, 4, 5, 6, 7, 8, 9,
                    10 ** 1, 20, 30, 40, 50, 60, 70, 80, 90,
                    10 ** 2, 200, 300, 400, 500, 600, 700, 800, 900,
                    10 ** 3, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                    10 ** 4])
        plt.yticks([10 ** 0, 2, 3, 4, 5, 6, 7, 8, 9,
                    10 ** 1, 20, 30, 40, 50, 60, 70, 80, 90,
                    10 ** 2, 200, 300, 400, 500, 600, 700, 800, 900,
                    10 ** 3])
        '''

        plt.xlabel("Degree")

        plt.tight_layout()

        image_path = os.path.join(images_folder, fn)
        ax = plt.gca()
        #handles, labels = ax.get_legend_handles_labels()
        #ax.get_legend().remove()
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.savefig(image_path)

        plt.show()

        plt.clf()

    if distributions[3] is not None:
        label = "User"
        fn = "degree_distributions_bottom"

        if population is not None:
            label += "_{}".format(population)
            fn += "_{}".format(population)

        fn += ".pdf"

        plt.figure()

        if color is None:
            sns.scatterplot(x=distributions[2], y=distributions[3], label=label, legend=None, s=100, linewidth=0.25)
        else:
            hue_tmp = np.ones(len(distributions[2]))
            sns.scatterplot(x=distributions[2], y=distributions[3], hue=hue_tmp, s=100, linewidth=0.25, palette=[color])

            plt.legend(labels=['$U_{}$'.format(population+1)])

        plt.xscale("log")
        plt.yscale("log")

        '''
        plt.ylim(bottom=0.7, top=10 ** 4)
        plt.xlim(left=0.7, right=10 ** 4)

        plt.yticks([10 ** 0, 2, 3, 4, 5, 6, 7, 8, 9,
                    10 ** 1, 20, 30, 40, 50, 60, 70, 80, 90,
                    10 ** 2, 200, 300, 400, 500, 600, 700, 800, 900,
                    10 ** 3, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                    10 ** 4])

        plt.xticks([10 ** 0, 2, 3, 4, 5, 6, 7, 8, 9,
                    10 ** 1, 20, 30, 40, 50, 60, 70, 80, 90,
                    10 ** 2, 200, 300, 400, 500, 600, 700, 800, 900,
                    10 ** 3, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000,
                    10 ** 4])
        '''

        plt.xlabel("Degree")

        plt.tight_layout()

        image_path = os.path.join(images_folder, fn)
        ax = plt.gca()
        #handles, labels = ax.get_legend_handles_labels()
        #ax.get_legend().remove()
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

        plt.savefig(image_path)

        plt.show()

        plt.clf()


    #plt.figure()


def compute_degree_distributions(data):

    bottom_grouped_df = data.groupby("User")  # users
    top_grouped_df = data.groupby("Item")  # items

    user_items = []
    item_users = []

    for _, group in bottom_grouped_df:
        user_items.append(len(group))

    for _, group in top_grouped_df:
        item_users.append(len(group))

    bottom_x, bottom_distribution = np.unique(user_items, return_counts=True)
    top_x, top_distribution = np.unique(item_users, return_counts=True)

    #np.savetxt(os.path.join(output_folder, f"{dataset}_users_distribution.npy"), bottom_distribution)
    #np.savetxt(os.path.join(output_folder, f"{dataset}_items_distribution.npy"), top_distribution)

    return top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users

def plot_category_percentages(percentages, saving_path, populations, eps=None):
    params = {
        'figure.dpi': 200,
        "text.usetex": True,
        'legend.title_fontsize': 14,
        'legend.fontsize': 14,
        'axes.labelsize': 25,
        'axes.titlesize': 16,
        'xtick.labelsize': 32,
        'ytick.labelsize': 32,
        "font.family": "sans-serif",
        "font.sans-serif": "Computer Modern Sans Serif",
    }

    import matplotlib as mpl
    mpl.rcParams.update(params)

    percentages = np.array(percentages)

    colors = ["green", "blue", "red"]
    palette = {}

    for i, pop in enumerate(populations):
        palette[pop] = colors[i]

    fig, ax = plt.subplots()

    if eps is not None:
        plt.title("$\epsilon = {}$".format(eps), fontsize=30)
    sns.histplot(x=percentages[:, 0].astype(np.float32), hue=percentages[:, 1].astype(np.int32), palette=palette,
                      bins=20, kde=False, stat="probability", ax=ax)

    ax.set_ylim([0.0, 0.5])

    plt.ylabel("Proportion")

    '''
    legend = ax.get_legend()
    handles = legend.legend_handles
    legend_size = 10

    fig_legend, axi = plt.subplots(figsize=(1.7, .4))

    fig_legend.legend(handles, ["$U_1$", "$U_2$"], ncol=2, fontsize=legend_size)
    axi.axis('off')
    fig_legend.tight_layout()
    fig_legend.savefig("legend_u1_u2.pdf", dpi=200)
    '''

    ax.get_legend().remove()

    ax = plt.gca()
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    plt.tight_layout()

    #plt.xlim((-0.1, 1.1))
    plt.savefig(saving_path + "category_percentage_distribution.pdf")
    #plt.show()

def mu_sigma_to_alpha_beta(mu, sigma):
    """ For Chaney's custom Beta' function, we convert
        a mean and variance to an alpha and beta parameter
        of a Beta function. See footnote 3 page 3 of Chaney
        et al. for details.
    """
    alpha = ((1-mu) / (sigma**2) - (1/mu)) * mu**2
    beta = alpha * (1/mu - 1)
    return alpha, beta


def main(dataset, save_file=True):

    with open("config/{}.json".format(dataset)) as fp:
        params = json.load(fp)

    #sys.argv = args[:1]
    parser = argparse.ArgumentParser()
    t_args = argparse.Namespace()
    t_args.__dict__.update(params)
    params = parser.parse_args(args=None, namespace=t_args)

    rng = np.random.default_rng(12121995)

    σ = 1e-5
    num_users = params.num_users
    num_items = params.num_items
    num_attrs = 64

    pi = params.pi
    pi_xmax = params.pi_xmax
    pi_xmin = params.pi_xmin
    theta_first_param = params.theta_first_param
    theta_second_param = params.theta_second_param
    zeta = params.zeta

    psi = params.psi
    psi_xmax = params.psi_xmax
    psi_xmin = params.psi_xmin
    vartheta_first_param = params.vartheta_first_param
    vartheta_second_param = params.vartheta_second_param
    xi = params.xi

    Lambda = params.Lambda
    delta = params.delta
    tau = params.tau

    if "noisy" in params.dataset_name:
        dataset_path = 'data_generation_files/{}_{}/'.format(params.dataset_name, delta)
    else:
        dataset_path = 'data_generation_files/{}/'.format(params.dataset_name)

    ETA_users = [float(x) for x in params.ETA_users.split(",")]
    ETA_items = [float(x) for x in params.ETA_items.split(",")]

    EPSILON = params.EPSILON

    dataset_path += '{}_{}_{}_{}_{}_{}_{}_{}_{}_zeta_{}_xi_{}_lambda_{}_epsilon_{}_noise_{}/'.format(num_attrs, num_users,
                                            num_items, pi, theta_first_param, theta_second_param,
                                            psi, vartheta_first_param, vartheta_second_param, zeta, xi,
                                            Lambda, EPSILON, delta)

    print(dataset_path)

    num_populations = len(ETA_users)

    num_users_populations = [int(num_users * ETA_users[i]) for i in range(len(ETA_users))]

    users_populations_mapping = {}
    users = np.array(list(range(num_users)))
    old_n = 0
    for i, n in enumerate(num_users_populations):
        users_population_mapping_temp = dict.fromkeys(users[old_n:old_n+n], i)
        users_populations_mapping.update(users_population_mapping_temp)
        old_n += n


    if not exists(dataset_path):
        os.makedirs(dataset_path)

    # USERS

    ρ = []

    old_n = 0
    for i in range(num_populations):
        rng = np.random.default_rng(12121995)
        params = np.ones(num_attrs)
        if ETA_users[i] != 1.0:
            params[(num_attrs//num_populations)*i:(num_attrs//num_populations)*(i+1)] = EPSILON
        μ_ρ = rng.dirichlet(params, size=num_users_populations[i]) * 10
        ρ_subpop = [rng.dirichlet(p) for p in μ_ρ]
        #ρ[:num_users_non_rad] = ρ_subpop
        ρ.append(ρ_subpop)

    ρ = np.concatenate(ρ)

    #print("HERE")

    num_categories = len(ETA_items)
    num_items_categories = [int(num_items * ETA_items[i]) for i in range(len(ETA_items))]

    items_categories_mapping = {}
    items = np.array(list(range(num_items)))
    old_n = 0
    for i, n in enumerate(num_items_categories):
        items_categories_mapping_temp = dict.fromkeys(items[old_n:old_n+n], i)
        items_categories_mapping.update(items_categories_mapping_temp)
        old_n += n

    α = []

    for i in range(num_categories):
        rng = np.random.default_rng(12121995)
        params = np.ones(num_attrs)*100

        if ETA_items[i] != 1.0:
            params[(num_attrs//num_categories)*i:(num_attrs//num_categories)*(i+1)] = EPSILON

        μ_ρ = rng.dirichlet(params, size=num_items_categories[i]) * 0.1
        α_subcategory = [rng.dirichlet(p) for p in μ_ρ]

        α.append(np.array(α_subcategory))

    α = np.concatenate(α)

    rng = np.random.default_rng(12121995)
    np.random.seed(12121995)

    if pi == "power_law":
        dist = powerlaw.Power_Law(parameters=[theta_first_param], xmin=pi_xmin)
    elif pi == "power_law_with_cutoff":
        dist = powerlaw.Truncated_Power_Law(parameters=[theta_first_param, theta_second_param], xmin=pi_xmin)
    elif pi == "stretched_exponential":
        dist = powerlaw.Stretched_Exponential(parameters=[theta_first_param, theta_second_param], xmin=pi_xmin)
    elif pi == "log_normal":
        dist = powerlaw.Lognormal(parameters=[theta_first_param, theta_second_param], xmin=pi_xmin)
    elif pi == "exponential":
        dist = powerlaw.Exponential(parameters=[theta_first_param], xmin=pi_xmin)

    z = dist.generate_random(num_users)
    for i in range(len(z)):
        iter = 0
        while z[i] > num_items or (pi_xmax != -1 and z[i] > pi_xmax):
            np.random.seed(12121995 + i + iter)
            z[i] = dist.generate_random(1)[0]

            iter += 1
            if iter == 50:
                raise RuntimeError("Maximum iterations")

    p_u = (z / num_items) ** zeta
    p_u = np.repeat(p_u, num_items).reshape(num_users, num_items)

    np.random.seed(12121995)
    if psi == "power_law":
        dist = powerlaw.Power_Law(parameters=[vartheta_first_param], xmin=psi_xmin)
    elif psi == "power_law_with_cutoff":
        dist = powerlaw.Truncated_Power_Law(parameters=[vartheta_first_param, vartheta_second_param], xmin=psi_xmin)
    elif psi == "stretched_exponential":
        dist = powerlaw.Stretched_Exponential(parameters=[vartheta_first_param, vartheta_second_param], xmin=psi_xmin)
    elif psi == "log_normal":
        dist = powerlaw.Lognormal(parameters=[vartheta_first_param, vartheta_second_param], xmin=psi_xmin)
    elif psi == "exponential":
        dist = powerlaw.Exponential(parameters=[vartheta_first_param], xmin=psi_xmin)

    y = dist.generate_random(num_items)
    for i in range(len(y)):
        iter = 0
        while y[i] > num_users or (psi_xmax != -1 and y[i] > psi_xmax):
            np.random.seed(12121995 + i + iter)
            y[i] = dist.generate_random(1)[0]

            iter += 1
            if iter == 50:
                raise RuntimeError("Maximum iterations")

    p_i = (y / num_users) ** xi
    p_i = np.repeat(p_i, num_users).reshape(num_items, num_users).T

    mu_eta = 0.98
    eta_alphas, eta_betas = mu_sigma_to_alpha_beta(mu_eta, σ)
    ω = rng.beta(eta_alphas, eta_betas, size=(num_users, num_items))

    true_T = ρ @ α.T
    noisy_T = delta * true_T + (1 - delta) * ω

    T = noisy_T * (p_u * p_i * Lambda)

    T[T > 1] = 0.99
    ρ_α = np.clip(T, 1e-9, None)

    a, b = mu_sigma_to_alpha_beta(ρ_α, σ)

    T = rng.beta(a, b, size=(num_users, num_items))

    ETA_users_str = str(ETA_users).replace("[", "").replace("]", "").replace(", ", "_")
    ETA_items_str = str(ETA_items).replace("[", "").replace("]", "").replace(", ", "_")

    saving_path = dataset_path + f"Populations_{ETA_users_str}_Categories_{ETA_items_str}/"

    if not exists(saving_path):
        os.makedirs(saving_path)


    def generate_history(user, p):
        items = []

        rng_2 = np.random.default_rng(user)

        iter = 0

        while len(items) < tau:
            for item, p_i in enumerate(p):
                if item not in items and rng_2.binomial(1, p_i):
                    items.append(item)

            iter += 1
            if iter == 5000:
                raise RuntimeError("Maximum iterations")

        if user % 500 == 0:
            print("User:", user, len(items))

        return items


    histories = Parallel(n_jobs=10, prefer="processes")(
        delayed(generate_history)(user, p) for user, p in enumerate(T))

    interactions = []

    for user, history in enumerate(histories):

        count = 0

        for item in history:
            interactions.append((user, item, users_populations_mapping[user], items_categories_mapping[item]))
            if items_categories_mapping[item] == 0:
                count += 1

    df = pd.DataFrame(
        columns=["User", "Item", "Population", "Category"], data=interactions)


    print(f"Num users {len(df['User'].unique())}")
    print(f"Num items {len(df['Item'].unique())}")
    print(f"Num interactions {df.shape[0]}")
    if (len(df['User'].unique()) * len(df['Item'].unique())) > 0:
        print(f"Density {df.shape[0] / (len(df['User'].unique()) * len(df['Item'].unique()))}")

        print(saving_path)
        show_interactions_plot(df, saving_path, EPSILON)

        top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users = compute_degree_distributions(
            df)

        plot_degree_distributions([top_x, top_distribution, bottom_x, bottom_distribution], saving_path)

        if len(ETA_users) > 1:
            for pop in range(len(ETA_users)):
                temp_df = df[df['Population'] == pop]
                top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users = compute_degree_distributions(temp_df)
                plot_degree_distributions([None, None, bottom_x, bottom_distribution], saving_path, population=pop, color=sns.color_palette("tab10")[1])

        if len(ETA_items) > 1:
            for cat in range(len(ETA_items)):
                temp_df = df[df['Category'] == cat]
                top_x, top_distribution, bottom_x, bottom_distribution, user_items, item_users = compute_degree_distributions(
                    temp_df)
                plot_degree_distributions([top_x, top_distribution, None, None], saving_path, category=cat, color=sns.color_palette("tab10")[0])

                if cat == 0 and len(ETA_items) > 1:
                    plot_category_percentages(percentages, saving_path, np.arange(len(ETA_users)), eps=EPSILON)


    if save_file:
        if "noisy" in dataset:
            df.to_csv(saving_path +
                      '{}_{}.tsv'.format(dataset, delta), index=False, sep="\t")
        else:
            df.to_csv(saving_path + '{}.tsv'.format(dataset), index=False, sep="\t")

    print("Finished")

if __name__ == '__main__':
    dataset = "movielens-1m_synthetic"
    main(dataset)

