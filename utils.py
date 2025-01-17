import torch
import random
import numpy as np
import pickle
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_scipy_sparse_matrix

truncate_len = 30
truncate_len_i = 30
soc_len = 30
cor_len = 30


def collate_fn(batch_data):
    uids, iids, ratings, tids = [], [], [], []
    u_items, u_users, i_users, i_items = [], [], [], []
    u_items_len, u_users_len, i_users_len, i_items_len = [], [], [], []
    soc_edge_index = []
    inter_edge_index = []

    # Create a mapping from global node ids to local node ids
    user_to_local_id = {}
    item_to_local_id = {}
    user_current_id = 0
    item_current_id = len(batch_data)

    for idx, data in enumerate(batch_data):
        uid = data['uids']
        uids.append(uid)
        iid = data['iids']
        iids.append(iid)
        ratings.append(data['ratings'])
        tids.append(data['tids'])

        u_items_data = data['u_items']
        u_users_data = data['u_users']
        i_users_data = data['i_users']
        i_items_data = data['i_items']

        # user-items
        if len(u_items_data) <= truncate_len:
            temp = np.array(u_items_data)
            temp = temp[np.lexsort(temp.T)].tolist()
            u_items.append(temp)
        else:
            temp = np.array(random.sample(u_items_data, truncate_len))
            temp = temp[np.lexsort(temp.T)].tolist()
            u_items.append(temp)
        u_items_len.append(min(len(u_items_data), truncate_len))

        # user-users
        if len(u_users_data) < soc_len:
            tmp_users = [item for item in u_users_data]
            tmp_users.append(uid)
            u_users.append(tmp_users)
        else:
            sample_index = random.sample(list(range(len(u_users_data))), soc_len - 1)
            tmp_users = [u_users_data[si] for si in sample_index]
            tmp_users.append(uid)
            u_users.append(tmp_users)

        u_users_len.append(min(len(u_users_data) + 1, soc_len))

        # item-users
        if len(i_users_data) <= truncate_len_i:
            temp = np.array(i_users_data)
            temp = temp[np.lexsort(temp.T)].tolist()
            i_users.append(temp)
        else:
            temp = np.array(random.sample(i_users_data, truncate_len_i))
            temp = temp[np.lexsort(temp.T)].tolist()
            i_users.append(temp)
        i_users_len.append(min(len(i_users_data), truncate_len_i))

        if len(i_items_data) < cor_len:
            tmp_items = [item for item in i_items_data]
            tmp_items.append(iid)
            i_items.append(tmp_items)
        else:
            sample_index = random.sample(list(range(len(i_items_data))), cor_len - 1)
            tmp_items = [i_items_data[si] for si in sample_index]
            tmp_items.append(iid)
            i_items.append(tmp_items)

        i_items_len.append(min(len(i_items_data) + 1, cor_len))

        # Establish indices for users and items.
        if (uid, idx) not in user_to_local_id:
            user_to_local_id[(uid, idx)] = user_current_id
            user_current_id += 1
        if (iid, idx) not in item_to_local_id:
            item_to_local_id[(iid, idx)] = item_current_id
            item_current_id += 1

    # Construct edge indices.
    for idx, data in enumerate(batch_data):
        uid = data['uids']
        iid = data['iids']
        # Social edges between users and friends.
        for u_user in data['u_users']:
            # Ensure that friend nodes are in the current batch.
            for friend_idx in range(len(batch_data)):
                if (u_user, friend_idx) in user_to_local_id:
                    soc_edge_index.append([user_to_local_id[(uid, idx)], user_to_local_id[(u_user, friend_idx)]])
                    soc_edge_index.append([user_to_local_id[(u_user, friend_idx)], user_to_local_id[(uid, idx)]])

        # interaction graph edge
        i_users_data = data['i_users']
        for i_user in i_users_data:
            for item_idx in range(len(batch_data)):
                if (i_user[0], item_idx) in user_to_local_id:
                    inter_edge_index.append([item_to_local_id[(iid, idx)], user_to_local_id[(i_user[0], item_idx)]])
                    inter_edge_index.append([user_to_local_id[(i_user[0], item_idx)], item_to_local_id[(iid, idx)]])

    batch_size = len(batch_data)

    # Convert edge indices to a Tensor.
    soc_edge_index = torch.tensor(soc_edge_index, dtype=torch.long).t().contiguous()
    inter_edge_index = torch.tensor(inter_edge_index, dtype=torch.long).t().contiguous()

    batch_size = len(batch_data)
    u_items_maxlen = max([len(u) for u in u_items])
    u_users_maxlen = max([len(u) for u in u_users])
    i_users_maxlen = max([len(i) for i in i_users])
    i_items_maxlen = max([len(i) for i in i_items])

    u_item_pad = torch.zeros([batch_size, u_items_maxlen, 3], dtype=torch.long)
    for i, ui in enumerate(u_items):
        u_item_pad[i, :len(ui), :] = torch.LongTensor(ui)

    u_user_pad = torch.zeros([batch_size, u_users_maxlen], dtype=torch.long)
    for i, uu in enumerate(u_users):
        u_user_pad[i, :len(uu)] = torch.LongTensor(uu)

    i_user_pad = torch.zeros([batch_size, i_users_maxlen, 3], dtype=torch.long)
    for i, iu in enumerate(i_users):
        i_user_pad[i, :len(iu), :] = torch.LongTensor(iu)

    i_item_pad = torch.zeros([batch_size, i_items_maxlen], dtype=torch.long)
    for i, ii in enumerate(i_items):
        i_item_pad[i, :len(ii)] = torch.LongTensor(ii)

    adj_matrix = torch.zeros((batch_size * 2, batch_size * 2), dtype=torch.float32)

    for idx in range(inter_edge_index.size(1)):
        i, j = inter_edge_index[:, idx]
        adj_matrix[i, j] = 1
        adj_matrix[j, i] = 1  # Symmetry

    return torch.LongTensor(uids), torch.LongTensor(iids), torch.FloatTensor(ratings), torch.IntTensor(
        tids), u_item_pad, u_user_pad, i_user_pad, i_item_pad, soc_edge_index, adj_matrix


def load_data(dataset):
    print('Loading data...')
    with open(f'datasets/{dataset}/dataset_filter5.pkl', 'rb') as f:
        train_set = pickle.load(f)
        valid_set = pickle.load(f)
        test_set = pickle.load(f)

    with open(f'datasets/{dataset}/list_filter5.pkl', 'rb') as f:
        u_items_list = pickle.load(f)
        u_users_list = pickle.load(f)
        u_users_items_list = pickle.load(f)
        i_users_list = pickle.load(f)
        i_items_list = pickle.load(f)
        i_items_users_list = pickle.load(f)
        user_count, item_count, rate_count, time_count = pickle.load(f)

    return train_set, valid_set, test_set, u_items_list, u_users_list, u_users_items_list, i_users_list, i_items_list, i_items_users_list, user_count, item_count, rate_count, time_count
