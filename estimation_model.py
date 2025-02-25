import torch
import torch.nn as nn
import torch.nn.functional as F

class ProbabilityModel(nn.Module):
    def __init__(self, p=1, name="ProbabilityDistribution"):
        super(ProbabilityModel, self).__init__()
        self.p = p
        self.name = name

    def forward(self, x):
        pass

    def get_params(self):
        pass


class PowerLaw(ProbabilityModel):
    def __init__(self, p=1, xmin=1, seed=1, device="cpu"):
        super(PowerLaw, self).__init__(p, "PowerLaw")

        self.xmin = torch.FloatTensor([xmin]).to(device)
        self.device = device

        self.log_alpha = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 2.0, 3.0), requires_grad=True).to(device)


    def forward(self, x):
        '''
        returns log prob(x|alpha)
        '''
        eps = 1e-4

        # \alpha needs to be > 1

        alpha = torch.exp(self.log_alpha)

        f = -alpha * torch.log(x)
        C = (alpha-1) * torch.log(self.xmin) + torch.log(alpha-1+eps)

        return f + C

    def get_params(self):
        return ('Alpha', torch.exp(self.log_alpha).item())


class PowerLawWithCutOff(ProbabilityModel):
    def __init__(self, p=1, xmin=1, seed=1):
        super(PowerLawWithCutOff, self).__init__(p, "PowerLawWithCutOff")

        #torch.manual_seed(seed)

        self.xmin = torch.FloatTensor([xmin])

        self.logit_alpha = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.1, 0.5), requires_grad=True)
        self.log_Lambda = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.1, 0.5), requires_grad=True)

    def forward(self, x):
        '''
        #returns log prob(x|alpha)
        '''

        eps = 1e-5

        Lambda = torch.exp(self.log_Lambda)

        alpha = F.sigmoid(self.logit_alpha)

        f = torch.log(x)*(-alpha) - Lambda * x

        C = (torch.tensor(1.) - alpha) * torch.log(Lambda) - torch.log(torch.special.gammainc(torch.tensor(1.) - alpha, Lambda * self.xmin))

        return f + C

    def get_params(self):
        return ('Alpha', F.sigmoid(self.logit_alpha).item(), 'Lambda', torch.exp(self.log_Lambda).item())


class Lognormal(ProbabilityModel):
    def __init__(self, p=1, xmin=1, seed=1, device="cpu"):
        super(Lognormal, self).__init__(p, "LogNormal")

        self.xmin = xmin.to(device)

        self.device = device

        self.mu = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.1, 2.0), requires_grad=True).to(device)
        self.sigma = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.1, 2.0), requires_grad=True).to(device)


    def forward(self, x):
        '''
        returns log prob(x|alpha)
        '''

        f = -torch.log(x) - ((torch.log(x) - self.mu) ** 2) / (2 * self.sigma ** 2)

        from scipy.constants import pi
        pi_torch = torch.tensor(pi).to(self.device)
        constant = torch.tensor(2.).to(self.device)

        C = + 0.5 * (torch.log(constant) - torch.log(pi_torch) - constant * torch.log(self.sigma)) + \
            torch.log(torch.erfc((torch.log(self.xmin) - self.mu) / (torch.sqrt(constant) * self.sigma)))


        return f + C

    def get_params(self):
        return ('Mu', self.mu.item(), 'Sigma', self.sigma.item())


class StretchedExponential(ProbabilityModel):
    def __init__(self, p=1, xmin=1, seed=1, device="cpu"):
        super(StretchedExponential, self).__init__(p, "StretchedExponential")

        #torch.manual_seed(seed)

        self.xmin = xmin.to(device)

        self.device = device

        self.log_Lambda = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.01, 0.2), requires_grad=True).to(device)
        self.log_beta = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.01, 0.2), requires_grad=True).to(device)

    def forward(self, x):
        '''
        returns log prob(x|alpha)
        '''

        # \lambda and \beta need to be > 0

        # print("Log lambda", self.log_Lambda)

        Lambda = torch.exp(self.log_Lambda)
        beta = torch.exp(self.log_beta)
        # print("Lambda", Lambda)

        f = (beta - 1) * (torch.log(x) + torch.log(Lambda)) - ((Lambda * x) ** beta)
        # print("f", f)

        C = torch.log(beta) + torch.log(Lambda) + ((Lambda * self.xmin) ** beta)
        # print("C", C)

        return f + C

    def get_params(self):
        return ('Lambda', torch.exp(self.log_Lambda).item(), 'Beta', torch.exp(self.log_beta).item())


class Exponential(ProbabilityModel):
    def __init__(self, p=1, xmin=1, seed=1, device="cpu"):
        super(Exponential, self).__init__(p, "Exponential")

        self.xmin = xmin.to(device)

        self.device = device

        self.log_Lambda = nn.Parameter(torch.nn.init.uniform_(torch.empty(1), 0.01, 0.2), requires_grad=True).to(device)

    def forward(self, x):
        '''
        returns log prob(x|alpha)
        '''

        # \lambda needs to be > 0

        Lambda = torch.exp(self.log_Lambda)

        f = -(Lambda * x)

        C = torch.log(Lambda) + (Lambda * self.xmin)

        return f + C

    def get_params(self):
        return ('Lambda', torch.exp(self.log_Lambda).item())


class Q_phi_Network(nn.Module):
    def __init__(self, hidden_dim_users_items, num_users, num_items, probs_user, probs_item, batch_size, device):
        super(Q_phi_Network, self).__init__()

        self.batch_size = batch_size
        self.device = device

        self.emb_users = nn.Embedding(num_users, hidden_dim_users_items)
        self.emb_items = nn.Embedding(num_items, hidden_dim_users_items)

        self.emb_theta_users = nn.Embedding(num_users, 1)
        self.emb_phi_items = nn.Embedding(num_items, 1)

        self.sigma_u = torch.tensor(0.01)
        self.sigma_i = torch.tensor(0.01)

        K = len(probs_user)
        H = len(probs_item)

        self.logit_pi = nn.Parameter(torch.nn.init.uniform_(torch.empty(K), 0.1, 1.0), requires_grad=True)
        self.logit_psi = nn.Parameter(torch.nn.init.uniform_(torch.empty(H), 0.1, 1.0), requires_grad=True)

        self.mlp_density = nn.Sequential(nn.Linear(1, 10),
                                         nn.ReLU(),
                                         nn.Linear(10, 10),
                                         nn.ReLU(),
                                         nn.Linear(10, 1))

        self.probs_user = probs_user
        self.probs_item = probs_item


    def compute_log_probabilities(self, x, y):
        K, H = len(self.probs_user), len(self.probs_item)

        log_probs_p = torch.zeros(x.shape[0], K).to(self.device)
        log_probs_q = torch.zeros(y.shape[0], H).to(self.device)

        for k in range(K):
            log_probs_p[:, k] = self.probs_user[k](x)

            # Debug
            if len(log_probs_p[:, k][log_probs_p[:, k].isnan()]) > 0:
                print("NAN USERS", self.probs_user[k].name, self.probs_user[k].get_params())

        for h in range(H):
            log_probs_q[:, h] = self.probs_item[h](y)

            # Debug
            if len(log_probs_q[:, h][log_probs_q[:, h].isnan()]) > 0:
                print("NAN ITEMS", self.probs_item[h].name, self.probs_item[h].get_params())

        return log_probs_p, log_probs_q


    def forward(self, users, items, users_degree, items_degree, density):

        rho_user = torch.softmax(self.emb_items(items), dim=-1)
        alpha_item = torch.softmax(self.emb_users(users), dim=-1)

        rho_alpha = torch.sum(rho_user * alpha_item, dim=-1) # Eq 6

        theta_user = torch.sigmoid(self.emb_theta_users(users)) # Eq 12
        phi_item = torch.sigmoid(self.emb_phi_items(items)) # Eq 12

        eta = self.mlp_density(density)
        lmbd = torch.exp(eta)

        pred_interactions = lmbd * rho_alpha * theta_user.reshape(-1) * phi_item.reshape(-1)

        log_p_user, log_q_item = self.compute_log_probabilities(users_degree[users], items_degree[items])

        return rho_user, alpha_item, log_p_user, log_q_item, \
            self.logit_pi, self.logit_psi, theta_user, phi_item, self.sigma_u, self.sigma_i, \
            pred_interactions


class Network_loss(nn.Module):
    def __init__(self, device):
        super(Network_loss, self).__init__()

        self.BCE_loss = nn.BCELoss()
        self.device = device

    def entropy(self, p, eps=1e-10):
        return -torch.sum(p * torch.log(p + eps), dim=0).mean()

    def log_dirichlet(self, p, a):
        value = torch.distributions.Dirichlet(a).log_prob(p)
        return value

    def log_beta_prime(self, p, mu, sigma):
        mu = torch.clamp(mu, min=0.001, max=0.999)

        a = mu * (mu * (1 - mu) / (sigma ** 2) - 1)
        b = a * (1 / mu - 1)

        value = torch.distributions.Beta(a, b).log_prob(p) - torch.log(torch.tensor(1000).to(self.device))
        return value

    def forward(self, params, users_degree_relative, items_degree_relative, ratings, objective):

        rho_user, alpha_item, log_p_user, log_q_item, \
            logit_pi, logit_psi, theta_user, phi_item, sigma_u, sigma_i, \
            pred_interactions = params

        eps = 1e-5

        # PREFERENCES
        pred_interactions = torch.clamp(pred_interactions, max=0.9999999)
        loss_preferences = self.BCE_loss(pred_interactions, ratings)

        # PRIORS

        # LongTail

        log_pi = F.log_softmax(logit_pi, dim=0)
        log_psi = F.log_softmax(logit_psi, dim=0)

        K = log_p_user.shape[1]
        H = log_q_item.shape[1]

        matrix_k_h = torch.zeros((K, H))
        for k in range(K):
            for h in range(H):
                numerator = log_pi[k] + log_psi[h] + log_p_user[:, k].sum() + log_q_item[:, h].sum()
                matrix_k_h[k, h] = numerator + eps

        softmax_matrix_k_h = (matrix_k_h / matrix_k_h.sum()).detach()

        log_probs_users = 0.0
        log_probs_items = 0.0
        for k in range(K):
            for h in range(H):
                log_probs_users += (log_p_user[:, k].sum() + log_pi[k]) * softmax_matrix_k_h[k, h]
                log_probs_items += (log_q_item[:, h].sum() + log_psi[h]) * softmax_matrix_k_h[k, h]

        # DIRICHLET
        mu_item = torch.ones_like(alpha_item) / alpha_item.shape[1]
        log_priors_dirichlet = self.log_dirichlet(alpha_item, mu_item * 0.1)

        # BETA
        log_priors_beta_user = self.log_beta_prime(theta_user.reshape(-1), users_degree_relative, sigma_u)
        log_priors_beta_item = self.log_beta_prime(phi_item.reshape(-1), items_degree_relative, sigma_i)

        if objective == "preferences":
            if (- log_priors_dirichlet.sum()) > 0.0:
                c1 = 0.0
                c2 = 1.0
                c3 = 1.0
                c4 = 1.0
            else:
                c1 = 0.0
                c2 = 0.0
                c3 = 1.0
                c4 = 1.0
        elif objective == "longtails":
            c1 = 1.0
            c2 = 0.0
            c3 = 0.0
            c4 = 0.0

        # LOSS
        loss = - (log_probs_users + log_probs_items) * c1 \
               - log_priors_dirichlet.sum() * c2 \
               + loss_preferences.sum() * c3 \
               - (log_priors_beta_user.sum() + log_priors_beta_item.sum()) * c4

        return loss, -log_probs_users.detach(), -log_probs_items.detach(), -log_priors_dirichlet.detach().sum(), \
            loss_preferences.detach().sum(), -(log_priors_beta_user.detach().sum() + log_priors_beta_item.detach().sum()),\
            softmax_matrix_k_h
