import os
import time
import json
import argparse
import pickle
import random
import numpy as np
from torch._C import device
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import collate_fn, load_data
from model import HR_HGDN
from dataloader import HGDNDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Ciao', help='dataset: Ciao/Epinions')
parser.add_argument('--batch_size', type=int, default=128, help='input batch size')
parser.add_argument('--embed_dim', type=int, default=128, help='the dimension of embedding')
parser.add_argument('--epoch', type=int, default=80, help='the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1234, help='the number of random seed to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--device', type=int, default=0, help='the index of GPU device (-1 for CPU)')
parser.add_argument('--test', action='store_true', help='test model')
parser.add_argument('--item_cl_weight', type=float, default=1.2, help='the weight of item_cl_loss')
parser.add_argument('--cl_weight', type=float, default=0.1, help='the weight of cl loss')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--patience', type=int, default=8, help='early stopping patience')
parser.add_argument('--num_layers', type=int, default=2, help='the number of layers of GNN')
parser.add_argument('--eps', type=float, default=0.10, help='the random noise')
parser.add_argument('--gat_weight', type=float, default=0.4, help='the gat weight')
args = parser.parse_args()
print(args)
print(torch.cuda.is_available)

device = (torch.device('cpu') if args.device < 0 else torch.device(f'cuda:{args.device}'))

torch.backends.cudnn.benchmark = True
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)

here = os.path.dirname(os.path.abspath(__file__))
fn = 'weights/' + args.dataset

if not os.path.exists(fn):
    os.mkdir(fn)


def main():
    train_set, valid_set, test_set, u_items_list, u_users_list, u_users_items_list, i_users_list, i_items_list, i_items_users_list, user_count, item_count, rate_count, time_count = load_data(
        args.dataset)

    train_data = HGDNDataset(train_set, u_items_list, u_users_list, i_users_list, i_items_list, user_count, item_count)
    valid_data = HGDNDataset(valid_set, u_items_list, u_users_list, i_users_list, i_items_list, user_count, item_count)
    test_data = HGDNDataset(test_set, u_items_list, u_users_list, i_users_list, i_items_list, user_count, item_count)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    model = HR_HGDN(num_users=user_count + 1, num_items=item_count + 1, num_rate_levels=rate_count + 1,
                    emb_dim=args.embed_dim, device=args.device).to(device)

    if args.test:
        print('Load checkpoint and testing...')
        ckpt = torch.load(f'{fn}/random_best_checkpoint.pth.tar', map_location=device)
        model.load_state_dict(ckpt['state_dict'])
        avg_mae, avg_rmse, avg_cl_loss = validate(test_loader, model)
        print(f"Test: MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, CL Loss: {avg_cl_loss:.4f}")
        return

    optimizer = optim.RMSprop(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.MSELoss()

    best_valid_loss = float('inf')
    best_epoch = 0
    valid_loss_list = []
    test_loss_list = []

    best_mae = float('inf')
    best_rmse = float('inf')
    patience_counter = 0

    for epoch in tqdm(range(args.epoch)):
        train_for_epoch(train_loader, model, optimizer, epoch, args.epoch, criterion)
        avg_mae, avg_rmse, avg_cl_loss = validate(valid_loader, model)
        valid_loss_list.append([avg_mae, avg_rmse, avg_cl_loss])
        test_mae, test_rmse, test_cl_loss = validate(test_loader, model)
        test_loss_list.append([test_mae, test_rmse, test_cl_loss])

        ckpt_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        current_valid_loss = avg_rmse

        if avg_rmse < best_rmse or avg_mae < best_mae:
            best_rmse = avg_rmse
            best_mae = avg_mae
            torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'{fn}/random_best_checkpoint.pth.tar')
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print("Early stopping triggered")
            break

        print(
            f'Epoch {epoch} validation: MAE: {avg_mae:.4f}, RMSE: {avg_rmse:.4f}, CL Loss: {avg_cl_loss:.4f}, Best MAE: {best_mae:.4f}, test_MAE: {test_mae:.4f}, test_RMSE: {test_rmse:.4f}, test_CL Loss: {test_cl_loss:.4f}')


def train_for_epoch(train_loader, model, optimizer, epoch, num_epochs, criterion):
    model.train()
    sum_epoch_loss = 0
    sum_cl_loss = 0
    sum_mae = 0
    sum_rmse = 0

    for i, (uids, iids, ratings, tids, u_item_pad, u_user_pad, i_user_pad, i_item_pad, soc_edge_index,
            inter_adj_matrix) in enumerate(
        train_loader):
        uids = uids.to(device)
        iids = iids.to(device)
        ratings = ratings.to(device)
        u_item_pad = u_item_pad.to(device)
        i_user_pad = i_user_pad.to(device)
        soc_edge_index = soc_edge_index.to(device)
        inter_adj_matrix = inter_adj_matrix.to(device)

        optimizer.zero_grad()
        preds, final_user_embedding, final_item_embedding, last_user_emb, last_item_emb = model(uids, iids,
                                                                                                u_item_pad,
                                                                                                i_user_pad,
                                                                                                soc_edge_index,
                                                                                                inter_adj_matrix,
                                                                                                perturbed=True,
                                                                                                num_layers=args.num_layers,
                                                                                                eps=args.eps,
																								gat_weight=args.gat_weight)

        rec_loss = criterion(preds, ratings)
        cl_loss = model.lightgcn.contrastive_loss(final_user_embedding, last_user_emb, final_item_embedding,
                                                  last_item_emb, args.item_cl_weight)

        total_loss = rec_loss + cl_loss * args.cl_weight
        total_loss.backward()
        optimizer.step()

        mae = torch.mean(torch.abs(preds - ratings))
        rmse = torch.sqrt(torch.mean((preds - ratings) ** 2))

        sum_epoch_loss += rec_loss.item()
        sum_cl_loss += cl_loss.item()
        sum_mae += mae.item()
        sum_rmse += rmse.item()

        if i % 100 == 0:
            mean_loss = sum_epoch_loss / (i + 1)
            mean_cl_loss = sum_cl_loss / (i + 1)
            mean_mae = sum_mae / (i + 1)
            mean_rmse = sum_rmse / (i + 1)
            print(
                f'[TRAIN] Epoch {epoch + 1}/{num_epochs}, Batch {i}, Loss: {total_loss.item():.4f}, CL Loss: {cl_loss.item():.4f}, MAE: {mae.item():.4f}, RMSE: {rmse.item():.4f}, Avg Loss: {mean_loss:.4f}, Avg CL Loss: {mean_cl_loss:.4f}, Avg MAE: {mean_mae:.4f}, Avg RMSE: {mean_rmse:.4f}')


def validate(valid_loader, model):
    model.eval()
    sum_mae = 0
    sum_rmse = 0
    sum_cl_loss = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for uids, iids, ratings, tids, u_item_pad, u_user_pad, i_user_pad, i_item_pad, soc_edge_index, inter_adj_matrix in valid_loader:
            uids = uids.to(device)
            iids = iids.to(device)
            ratings = ratings.to(device)
            u_item_pad = u_item_pad.to(device)
            i_user_pad = i_user_pad.to(device)
            soc_edge_index = soc_edge_index.to(device)
            inter_adj_matrix = inter_adj_matrix.to(device)

            preds, final_user_embedding, final_item_embedding, last_user_emb, last_item_emb = model(uids, iids,
                                                                                                    u_item_pad,
                                                                                                    i_user_pad,
                                                                                                    soc_edge_index,
                                                                                                    inter_adj_matrix,
                                                                                                    perturbed=True,
                                                                                                    num_layers=args.num_layers,
                                                                                                    eps=args.eps,
																									gat_weight=args.gat_weight)

            rec_loss = criterion(preds, ratings)
            cl_loss = model.lightgcn.contrastive_loss(final_user_embedding, last_user_emb, final_item_embedding,
                                                      last_item_emb, args.item_cl_weight)
            sum_cl_loss += cl_loss.item()

            mae = torch.mean(torch.abs(preds - ratings))
            rmse = torch.sqrt(torch.mean((preds - ratings) ** 2))
            sum_mae += mae.item()
            sum_rmse += rmse.item()

    avg_mae = sum_mae / len(valid_loader)
    avg_rmse = sum_rmse / len(valid_loader)
    avg_cl_loss = sum_cl_loss / len(valid_loader)
    return avg_mae, avg_rmse, avg_cl_loss


if __name__ == '__main__':
    main()
