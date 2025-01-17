import torch
from torch.utils.data import Dataset


class HGDNDataset(Dataset):
    def __init__(self, data, u_items_list, u_users_list, i_users_list, i_items_list, num_users, num_items):
        self.data = data
        self.u_items_list = u_items_list
        self.u_users_list = u_users_list
        self.i_users_list = i_users_list
        self.i_items_list = i_items_list
        self.num_users = num_users
        self.num_items = num_items

    def __getitem__(self, index):
        uid = self.data[index][0]
        iid = self.data[index][1]
        rating = self.data[index][2]
        tid = self.data[index][3]

        u_items = self.u_items_list[uid]
        u_users = self.u_users_list[uid]
        i_users = self.i_users_list[iid]
        i_items = self.i_items_list[iid]

        data = {
            'uids': uid,
            'iids': iid,
            'ratings': rating,
            'tids': tid,
            'u_items': u_items,
            'u_users': u_users,
            'i_users': i_users,
            'i_items': i_items,
            'u_items_len': len(u_items),
            'u_users_len': len(u_users),
            'i_users_len': len(i_users),
            'i_items_len': len(i_items),
        }

        return data

    def __len__(self):
        return len(self.data)
