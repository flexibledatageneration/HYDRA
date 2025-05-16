# HYDRA

#### This is the official repository of the paper "Flexible Generation of Preference Data for Recommendation Analysis" accepted at KDD 2025
HyDRA is a novel model for generating realistic and flexible synthetic preference data—crucial for simulating and
analyzing recommendation systems in controlled environments.
Building effective recommender systems requires high-quality synthetic data that mirrors real-world user behavior.
HyDRA delivers on this by modeling: (i) User-item interaction intensity, (ii) Item popularity dynamics, 
(iii) User engagement patterns.
It creates user communities that mimic social influence and shared item adoption, as well as simulate
popularity and engagement using mixtures of distributions, offering greater realism and diversity.

## Approach
HyDRA generates synthetic datasets by simulating:

1. **User-Item Matching** — Based on latent Dirichlet-distributed feature vectors.
2. **User Engagement** — Modeled with long-tail (e.g., power-law, exponential) distributions.
3. **Item Popularity** — Simulated similarly via customizable probabilistic priors.

## Usage

> Make sure you have Python 3.8+ and the required packages installed:  
> pip install -r requirements.txt

---

## 1. Generate Synthetic Dataset

```bash
python main.py <dataset_name>
```

- **`<dataset_name>`**  
  Name of the JSON file (without `.json`) in `config/`, e.g. `power_law_power_law`.  
  Reads `config/<dataset_name>.json`, generates users/items/interactions, and saves to  
  `data_generation_files/<dataset_name>_<params…>/`.

- **Config parameters** (inside `config/<dataset_name>.json`):
  ```jsonc
  {
    "dataset_name": "power_law_power_law",
  
    "pi": "power_law", // Power-law distribution for users
    "theta_first_param": 1.5, // User distribution parameter #1
    "theta_second_param": 0.0, // User distribution parameter #2
    "pi_xmax": -1, // Max value sampled from distribution pi
    "pi_xmin": 1, // Min value sampled from distribution pi
    "zeta": 1.2, // Variable for density manipulation
  
    "psi": "power_law", // Power-law distribution for users
    "vartheta_first_param": 1.5, // Item distribution parameter #1
    "vartheta_second_param": 0.0, // Item distribution parameter #2
    "psi_xmax": -1, // Max value sampled from distribution psi
    "psi_xmin": 1, // Min value sampled from distribution psi
    "xi": 1.2, // Variable for density manipulation
  
    "Lambda": 8000, // Variable for density manipulation
    "delta": 1.0, // Noisy level injection (1.0 means no noise)

    "num_users": 6000,
    "num_items": 5000,
    "num_attrs": 64,
    "tau": 1, // Min size for users histories

    "ETA_users": "1.0", // Single user population
    "ETA_items": "1.0", // Single item population
    "EPSILON": 0.02 // Intersection level between populations
  }
  ```

---

## 2. Estimate Distribution Parameters

```bash
python estimation.py \
  -d <dataset> \
  -o <objective> \
  -ud <user_dist> \
  -id <item_dist>
```

- **`-d`, `--dataset`**  
  Name of a dataset. Example:  
  - `movielens-1m` (automatically downloads from GroupLens)  
  - `amazon` (expects `real_data/amazon/...` to exist)

- **`-o`, `--objective`**  
  - `longtails` — fit user/item degree distributions  
  - `preferences` — fit rating/value distributions
  - `both` 

- **`-ud`, `--user_dist`** & **`-id`, `--item_dist`**  
  One of `PowerLaw`, `Exponential`, `StretchedExponential`, `Lognormal`.

- **Example:**
  ```bash
  python estimation.py \
    -d movielens-1m \
    -o longtails \
    -ud PowerLaw \
    -id Lognormal
  ```

- **Output:**
  - Estimated parameters printed to console  
  - (Optional) Saves parameters and plots to `estimation_files/<dataset>_<params>/`

---

## 3. Evaluate Recommender Systems

```bash
python recsys.py <dataset>
```

- **`<dataset>`**  
  E.g. **python recsys.py power_law_power_law**, after generation you might have:
  ```text
  data_generation_files/
    power_law_power_law
  ```

- **Built-in models from RecBole:**  
  - `Pop`
  - `ItemKNN`
  - `BPR`  
  - `NeuMF`  
  - `MultiVAE`

- **Output:**
  - A CSV of recall/Hit@{5,10,20} per model in `recbole/<dataset>/`
