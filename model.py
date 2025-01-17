import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, LightGCN
from torch_geometric.utils import add_self_loops

lstm_layer = 4


class HR_HGDN(nn.Module):
    def __init__(self, num_users, num_items, num_rate_levels, emb_dim, device):
        super(HR_HGDN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_rate_levels = num_rate_levels
        self.emb_dim = emb_dim
        self.device = device
        self.user_emb = nn.Embedding(self.num_users, self.emb_dim, padding_idx=0)
        self.item_emb = nn.Embedding(self.num_items, self.emb_dim, padding_idx=0)
        self.rate_emb = nn.Embedding(self.num_rate_levels, self.emb_dim, padding_idx=0)

        self.user_model = _UserModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb, self.device)
        self.item_model = _ItemModel(self.emb_dim, self.user_emb, self.item_emb, self.rate_emb, self.device)

        self.gat = GATModule(self.emb_dim, self.device)
        self.lightgcn = LightGCN()

        self.rate_pred = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2 * self.emb_dim, self.emb_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim, self.emb_dim // 4),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(self.emb_dim // 4, 1)
        )

    def forward(self, uids, iids, u_item_pad, i_user_pad, soc_edge_index, inter_adj_matrix, perturbed=True,
                num_layers=2, eps=0.15, gat_weight=0.2):
        # User Interest Embedding and Item Attractiveness Embedding
        user_interest_emb = self.user_model(uids, u_item_pad, soc_edge_index)
        item_attraction_emb = self.item_model(iids, i_user_pad)

        # To store the embeddings of each layer in a multi-layer convolutional network
        all_user_embeddings = [user_interest_emb]
        all_item_embeddings = [item_attraction_emb]

        # multi-layer convolutional
        for _ in range(num_layers):
            user_gat_emb = self.gat(all_user_embeddings[-1], soc_edge_index)
            user_gcn_emb, item_gcn_emb = self.lightgcn(all_user_embeddings[-1], all_item_embeddings[-1],
                                                       inter_adj_matrix, perturbed=perturbed,
                                                       eps=eps)
            user_fused_emb = gat_weight * user_gat_emb + (1-gat_weight) * user_gcn_emb
            all_user_embeddings.append(user_fused_emb)
            all_item_embeddings.append(item_gcn_emb)
            soc_edge_index = add_edges_based_on_similarity(user_fused_emb, soc_edge_index, similarity_threshold=0.7)

        final_user_emb = torch.mean(torch.stack(all_user_embeddings, dim=1), dim=1)
        final_item_emb = torch.mean(torch.stack(all_item_embeddings, dim=1), dim=1)
        
        preds = self.rate_pred(torch.cat([final_user_emb, final_item_emb], dim=1)).squeeze()

        return preds, final_user_emb, final_item_emb, all_user_embeddings[-1], all_item_embeddings[-1]


class GATModule(nn.Module):
    def __init__(self, emb_dim, device, gat_heads=8, gat_dropout=0.5):
        super(GATModule, self).__init__()
        self.gat1 = GATConv(in_channels=emb_dim, out_channels=emb_dim // gat_heads, heads=gat_heads,
                            dropout=gat_dropout)
        self.gat2 = GATConv(in_channels=emb_dim, out_channels=emb_dim, heads=1, dropout=gat_dropout)

    def forward(self, x, edge_index):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        x = self.gat1(x, edge_index)
        x = self.gat2(x, edge_index)
        return x


def add_edges_based_on_similarity(user_embeddings, edge_index, similarity_threshold=0.9):
    norm_embeddings = F.normalize(user_embeddings, p=2, dim=1)
    similarity_matrix = torch.mm(norm_embeddings, norm_embeddings.t())

    new_edges = (similarity_matrix > similarity_threshold).nonzero(as_tuple=False)
    if new_edges.size(0) == 0:
        return edge_index
    else:
        new_edges = new_edges[new_edges[:, 0] != new_edges[:, 1]]
    new_edge_index = torch.cat([edge_index, new_edges.t()], dim=1)

    return new_edge_index


class _UserModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb, device, gat_heads=8, gat_dropout=0.5):
        super(_UserModel, self).__init__()
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.emb_dim = emb_dim
        self.device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.g_v = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.user_items_att = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_items = _Aggregation(self.emb_dim, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, lstm_layer, batch_first=True)
        self.eps = 1e-10

    def forward(self, uids, u_item_pad, soc_edge_index):
        q_j = self.item_emb(u_item_pad[:, :, 0])
        mask_u = torch.where(u_item_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        e_ij = self.rate_emb(u_item_pad[:, :, 1])
        x_ij = self.g_v(torch.cat([q_j, e_ij], dim=2).view(-1, 2 * self.emb_dim)).view(
            q_j.size())
        p_i = self.user_emb(uids).unsqueeze(1).expand(-1, q_j.size(1), -1)
        p_i = mask_u.unsqueeze(2).expand_as(p_i) * p_i

        alpha = self.user_items_att(torch.cat([self.w1(x_ij), self.w1(p_i)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_u.size())
        alpha = torch.exp(alpha) * mask_u
        alpha = alpha / (torch.sum(alpha, 1).unsqueeze(1).expand_as(alpha) + self.eps)

        h_iL = self.aggre_items(torch.sum(alpha.unsqueeze(2).expand_as(x_ij) * x_ij, 1))

        lstm_ia, _ = self.lstm(x_ij)
        h_iS = lstm_ia[:, -1, :]

        h_iI = h_iL * h_iS
        h_iI = F.dropout(h_iI, 0.5, training=self.training)

        return h_iI


class _ItemModel(nn.Module):
    def __init__(self, emb_dim, user_emb, item_emb, rate_emb, device):
        super(_ItemModel, self).__init__()
        self.emb_dim = emb_dim
        self.user_emb = user_emb
        self.item_emb = item_emb
        self.rate_emb = rate_emb
        self.device = torch.device('cpu') if device < 0 else torch.device(f'cuda:{device}')

        self.w1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.g_u = _MultiLayerPercep(2 * self.emb_dim, self.emb_dim)
        self.item_users_att_i = _MultiLayerPercep(2 * self.emb_dim, 1)
        self.aggre_users_i = _Aggregation(self.emb_dim, self.emb_dim)
        self.lstm = nn.LSTM(self.emb_dim, self.emb_dim, lstm_layer, batch_first=True)
        self.eps = 1e-10

    def forward(self, iids, i_user_pad):
        p_i = self.user_emb(i_user_pad[:, :, 0])
        mask_i = torch.where(i_user_pad[:, :, 0] > 0, torch.tensor([1.], device=self.device),
                             torch.tensor([0.], device=self.device))
        e_ij = self.rate_emb(i_user_pad[:, :, 1])
        y_ji = self.g_u(torch.cat([p_i, e_ij], dim=2).view(-1, 2 * self.emb_dim)).view(p_i.size())
        q_j = mask_i.unsqueeze(2).expand_as(p_i) * self.item_emb(iids).unsqueeze(1).expand_as(p_i)

        miu = self.item_users_att_i(torch.cat([self.w1(y_ji), self.w1(q_j)], dim=2).view(-1, 2 * self.emb_dim)).view(
            mask_i.size())
        miu = torch.exp(miu) * mask_i
        miu = miu / (torch.sum(miu, 1).unsqueeze(1).expand_as(miu) + self.eps)
        z_jL = self.aggre_users_i(torch.sum(miu.unsqueeze(2).expand_as(y_ji) * self.w1(y_ji), 1))

        lstm_jt, _ = self.lstm(y_ji)
        z_jS = lstm_jt[:, -1, :]

        z_jA = z_jL * z_jS
        z_jA = F.dropout(z_jA, p=0.5, training=self.training)

        return z_jA


class LightGCN(nn.Module):
    def __init__(self, tau=0.10):
        super(LightGCN, self).__init__()
        self.tau = tau

    def forward(self, user_embeddings, item_embeddings, adj_matrix, perturbed=False, eps=0.1):
        all_embeddings = [torch.cat([user_embeddings, item_embeddings], dim=0)]
        ego_embeddings = torch.sparse.mm(adj_matrix, all_embeddings[-1])
        
        if perturbed:  # Add noise after the first layer
            random_noise = torch.randn_like(ego_embeddings)
            ego_embeddings += torch.sign(ego_embeddings) * F.normalize(random_noise, p=2, dim=1) * eps

        user_embeddings, item_embeddings = torch.split(ego_embeddings,
                                                       [user_embeddings.size(0), item_embeddings.size(0)])
        return user_embeddings, item_embeddings

    def contrastive_loss(self, final_user_embeddings, last_user_embeddings, final_item_embeddings,
                         last_item_embeddings, alpha):

        user_loss = InfoNCE(final_user_embeddings, last_user_embeddings, self.tau)
        item_loss = InfoNCE(final_item_embeddings, last_item_embeddings, self.tau)
        return user_loss + alpha * item_loss


def InfoNCE(anchor_embeddings, contrast_embeddings, temperature):
    anchor_norm = F.normalize(anchor_embeddings, p=2, dim=1)
    contrast_norm = F.normalize(contrast_embeddings, p=2, dim=1)
    similarities = torch.mm(anchor_norm, contrast_norm.t()) / temperature
    batch_size = anchor_embeddings.size(0)
    targets = torch.arange(batch_size).to(anchor_embeddings.device)
    log_prob = F.log_softmax(similarities, dim=1)
    log_prob_pos = log_prob.gather(1, targets.unsqueeze(1)).squeeze()
    loss = -log_prob_pos.mean()
    return loss


class _MultiLayerPercep(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_MultiLayerPercep, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2, bias=True),
            nn.LeakyReLU(0.2),
            nn.Linear(input_dim // 2, output_dim, bias=True),
        )

    def forward(self, x):
        return self.mlp(x)


class _Aggregation(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(_Aggregation, self).__init__()
        self.aggre = nn.Sequential(
            nn.Linear(input_dim, output_dim, bias=True),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.aggre(x)
