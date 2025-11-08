from model import Model
from utils import *
from sklearn.metrics import roc_auc_score
import random
import os
import dgl
import argparse
import copy
from tqdm import tqdm
import torch.nn.functional as F

# === ۱. ایمپورت توابع تقویت دینامیک از AD-GCL ===
from aug import get_sim, neighbor_pruning, neighbor_completion 

# برای مدیریت NumPy و PyTorch
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn 
# ================================================

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ['OMP_NUM_THREADS'] = '1'

parser = argparse.ArgumentParser(description='GRADATE_ADGCL')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--dataset', type=str, default='cora')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0.0)
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--embedding_dim', type=int, default=64)
parser.add_argument('--patience', type=int, default=100)
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--batch_size', type=int, default=300)
parser.add_argument('--subgraph_size', type=int, default=4)
parser.add_argument('--readout', type=str, default='avg')
parser.add_argument('--auc_test_rounds', type=int, default=256)
parser.add_argument('--negsamp_ratio_patch', type=int, default=6)
parser.add_argument('--negsamp_ratio_context', type=int, default=1)
parser.add_argument('--alpha', type=float, default=0.1, help='how much the first view involves')
parser.add_argument('--beta', type=float, default=0.1, help='how much the second view involves')

# === پارامترهای جدید AD-GCL ===
parser.add_argument('--W', type=int, default=5, help='Window size for ano_sim update')
parser.add_argument('--degree_threshold', type=int, default=8, help='Degree threshold for low-degree nodes')
parser.add_argument('--edge_mask_rate_1', type=float, default=0.1, help='Edge mask rate for view 1')
parser.add_argument('--edge_mask_rate_2', type=float, default=0.1, help='Edge mask rate for view 2')
parser.add_argument('--feat_drop_rate_1', type=float, default=0.1, help='Feature drop rate for view 1')
parser.add_argument('--feat_drop_rate_2', type=float, default=0.1, help='Feature drop rate for view 2')
# ==============================

args = parser.parse_args()

if __name__ == '__main__':

    print('Dataset: {}'.format(args.dataset), flush=True)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    for run in range(args.runs):

        seed = run + 1
        random.seed(seed)
        
        batch_size = args.batch_size
        subgraph_size = args.subgraph_size

        adj, features, labels, idx_train, idx_val,\
        idx_test, ano_label, str_ano_label, attr_ano_label = load_mat(args.dataset)

        features, _ = preprocess_features(features)
        
        dgl_graph = adj_to_dgl_graph(adj)

        nb_nodes = features.shape[0]
        ft_size = features.shape[1]
        nb_classes = labels.shape[1]

        # === ۲. حذف منطق تقویت ساختار ثابت GRADATE اصلی ===
        # adj_edge_modification = aug_random_edge(adj, 0.2)
        # adj = normalize_adj(adj)
        # adj = (adj + sp.eye(adj.shape[0])).todense()
        # adj_hat = normalize_adj(adj_edge_modification)
        # adj_hat = (adj_hat + sp.eye(adj_hat.shape[0])).todense()
        # =======================================================
        
        # === ۳. مقداردهی اولیه متغیرهای دینامیک AD-GCL ===
        adj_numpy = adj.todense()
        # degree: برای استفاده در pruning/completion
        degree = np.sum(adj_numpy, axis=1).flatten() 
        # node_dist: نمایش مجاورت (یا ماتریس گره به گره)
        node_dist = torch.FloatTensor(adj_numpy).to(device)
        
        # loss_matrix_list: برای ذخیره زیان گره در طول W دوره
        loss_matrix = np.zeros(nb_nodes)
        loss_matrix_list = []
        # ano_sim و sim: ماتریس‌های شباهت گره و ناهنجاری اولیه (همه یکسان)
        # توجه: اینها در طول آموزش به صورت دینامیک به روز می‌شوند.
        ano_sim = torch.ones(nb_nodes, nb_nodes).to(device)
        sim = torch.ones(nb_nodes, nb_nodes).to(device)
        # =======================================================

        # ویژگی‌ها را بدون بُعد اضافی به GPU می‌بریم (در داخل batch_idx به [np.newaxis] تبدیل می‌شود)
        features = torch.FloatTensor(features).to(device) 
        labels = torch.FloatTensor(labels[np.newaxis]).to(device)
        idx_train = torch.LongTensor(idx_train).to(device)
        idx_val = torch.LongTensor(idx_val).to(device)
        idx_test = torch.LongTensor(idx_test).to(device)

        all_auc = []


        print('\n# Run:{} with random seed:{}'.format(run, seed), flush=True)
        dgl.random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)

        model = Model(ft_size, args.embedding_dim, 'prelu', args.negsamp_ratio_patch, args.negsamp_ratio_context,
                      args.readout).to(device)
        optimiser = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        b_xent_patch = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_patch]).to(device))
        b_xent_context = nn.BCEWithLogitsLoss(reduction='none',
                                            pos_weight=torch.tensor([args.negsamp_ratio_context]).to(device))

        cnt_wait = 0
        best = 1e9
        best_t = 0
        batch_num = nb_nodes // batch_size + 1

        for epoch in range(args.num_epoch):

            model.train()

            # === A. به‌روزرسانی ano_sim (منطق AD-GCL) ===
            if epoch > 0 and (epoch % args.W == 0):
                if len(loss_matrix_list) > 0:
                    mean_loss_matrix = np.mean(loss_matrix_list, axis=0)
                    loss_matrix_list = [] # ریست کردن
                    
                    # محاسبه ano_sim
                    ano_sim_cpu = np.exp(mean_loss_matrix[:, np.newaxis] - mean_loss_matrix)
                    ano_sim = torch.FloatTensor(ano_sim_cpu).to(device)
                    
            # === B. تولید نماهای ساختاری دینامیک A1 و A2 در هر Epoch ===
            # View 1: هرس (Pruning) - برای GRADATE (adj)
            # توجه: توابع AD-GCL ویژگی‌های ویژگی (feat1, feat2) را نیز دراپ می‌کنند اما در GRADATE فقط از adj استفاده می‌کنیم.
            # ما فقط adj_view1_full را استخراج می‌کنیم.
            _, _, _, adj_view1_full = neighbor_pruning(
                dgl_graph, node_dist.cpu(), sim.cpu(), features.cpu(), 
                degree, args.feat_drop_rate_1, args.feat_drop_rate_1, args.degree_threshold
            )
            adj_view1_full = adj_view1_full.to(device)
            
            # View 2: تکمیل (Completion) - برای GRADATE (adj_hat)
            _, _, _, _, adj_view2_full, _ = neighbor_completion(
                dgl_graph, node_dist.cpu(), sim.cpu(), ano_sim.cpu(), features.cpu(), 
                degree, args.feat_drop_rate_1, args.edge_mask_rate_1, 
                args.feat_drop_rate_2, args.edge_mask_rate_2, args.degree_threshold, device
            )
            adj_view2_full = adj_view2_full.to(device)
            # ============================================================


            all_idx = list(range(nb_nodes))
            random.shuffle(all_idx)
            total_loss = 0.

            subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
            current_loss_matrix = np.zeros(nb_nodes) # ماتریس زیان موقت برای دوره جاری

            for batch_idx in range(batch_num):

                optimiser.zero_grad()

                is_final_batch = (batch_idx == (batch_num - 1))
                if not is_final_batch:
                    idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                else:
                    idx = all_idx[batch_idx * batch_size:]

                cur_batch_size = len(idx)

                lbl_patch = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_patch))), 1).to(device)

                lbl_context = torch.unsqueeze(torch.cat(
                    (torch.ones(cur_batch_size), torch.zeros(cur_batch_size * args.negsamp_ratio_context))), 1).to(device)

                ba = [] # نمای A1 (هرس)
                ba_hat = [] # نمای A2 (تکمیل)
                bf = []
                added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                added_adj_zero_col[:, -1, :] = 1.
                added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)

                for i in idx:
                    # === C. استخراج زیرگراف از نماهای دینامیک ===
                    # نمای A1 (هرس)
                    cur_adj = adj_view1_full[:, subgraphs[i], :][:, :, subgraphs[i]] 
                    # نمای A2 (تکمیل)
                    cur_adj_hat = adj_view2_full[:, subgraphs[i], :][:, :, subgraphs[i]]
                    # ویژگی‌ها
                    cur_feat = features[np.newaxis, subgraphs[i], :]
                    
                    ba.append(cur_adj)
                    ba_hat.append(cur_adj_hat)
                    bf.append(cur_feat)

                ba = torch.cat(ba)
                ba = torch.cat((ba, added_adj_zero_row), dim=1)
                ba = torch.cat((ba, added_adj_zero_col), dim=2)
                ba_hat = torch.cat(ba_hat)
                ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                bf = torch.cat(bf)
                bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)

                logits_1, logits_2, subgraph_embed, node_embed = model(bf, ba)
                logits_1_hat, logits_2_hat,  subgraph_embed_hat, node_embed_hat = model(bf, ba_hat)

                # === D. به‌روزرسانی sim و ذخیره امتیاز ناهنجاری ===
                # محاسبه sim (شباهت گره‌ای) برای استفاده در دوره بعدی
                # sim = get_sim(node_embed[:, -1, :], node_embed[:, -1, :]) # node_embed شامل گره‌های مثبت/منفی است.
                sim = get_sim(node_embed, node_embed)
                #subgraph-subgraph contrast loss (بدون تغییر)
                subgraph_embed = F.normalize(subgraph_embed, dim=1, p=2)
                subgraph_embed_hat = F.normalize(subgraph_embed_hat, dim=1, p=2)
                sim_matrix_one = torch.matmul(subgraph_embed, subgraph_embed_hat.t())
                sim_matrix_two = torch.matmul(subgraph_embed, subgraph_embed.t())
                sim_matrix_three = torch.matmul(subgraph_embed_hat, subgraph_embed_hat.t())
                temperature = 1.0
                sim_matrix_one_exp = torch.exp(sim_matrix_one / temperature)
                sim_matrix_two_exp = torch.exp(sim_matrix_two / temperature)
                sim_matrix_three_exp = torch.exp(sim_matrix_three / temperature)
                nega_list = np.arange(0, cur_batch_size - 1, 1)
                nega_list = np.insert(nega_list, 0, cur_batch_size - 1)
                sim_row_sum = sim_matrix_one_exp[:, nega_list] + sim_matrix_two_exp[:, nega_list] + sim_matrix_three_exp[:, nega_list]
                sim_row_sum = torch.diagonal(sim_row_sum)
                sim_diag = torch.diagonal(sim_matrix_one)
                sim_diag_exp = torch.exp(sim_diag / temperature)
                NCE_loss = -torch.log(sim_diag_exp / (sim_row_sum))
                NCE_loss = torch.mean(NCE_loss)


                loss_all_1 = b_xent_context(logits_1, lbl_context)
                loss_all_1_hat = b_xent_context(logits_1_hat, lbl_context)
                loss_1 = torch.mean(loss_all_1)
                loss_1_hat = torch.mean(loss_all_1_hat)

                loss_all_2 = b_xent_patch(logits_2, lbl_patch)
                loss_all_2_hat = b_xent_patch(logits_2_hat, lbl_patch)
                loss_2 = torch.mean(loss_all_2)
                loss_2_hat = torch.mean(loss_all_2_hat)

                loss_1 = args.alpha * loss_1 + (1 - args.alpha) * loss_1_hat #node-subgraph contrast loss
                loss_2 = args.alpha * loss_2 + (1 - args.alpha) * loss_2_hat #node-node contrast loss
                loss = args.beta * loss_1 + (1 - args.beta) * loss_2 + 0.1 * NCE_loss #total loss

                # محاسبه امتیاز ناهنجاری گره‌ای برای به‌روزرسانی ano_sim (منطق AD-GCL)
                with torch.no_grad():
                    test_logits_1 = torch.sigmoid(torch.squeeze(logits_1))
                    test_logits_2 = torch.sigmoid(torch.squeeze(logits_2))
                    test_logits_1_hat = torch.sigmoid(torch.squeeze(logits_1_hat))
                    test_logits_2_hat = torch.sigmoid(torch.squeeze(logits_2_hat))

                    # امتیاز ناهنجاری = منفی (امتیاز مثبت - میانگین امتیازات منفی)
                    ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(cur_batch_size, args.negsamp_ratio_context), dim=1))
                    ano_score_1_hat = - (test_logits_1_hat[:cur_batch_size] - torch.mean(test_logits_1_hat[cur_batch_size:].view(cur_batch_size, args.negsamp_ratio_context), dim=1))
                    ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(cur_batch_size, args.negsamp_ratio_patch), dim=1))
                    ano_score_2_hat = - (test_logits_2_hat[:cur_batch_size] - torch.mean(test_logits_2_hat[cur_batch_size:].view(cur_batch_size, args.negsamp_ratio_patch), dim=1))
                    
                    # ترکیب امتیازات
                    ano_score_batch = args.beta * (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_hat)  + \
                                      (1 - args.beta) * (args.alpha * ano_score_2 + (1 - args.alpha) * ano_score_2_hat)

                # ذخیره امتیاز ناهنجاری گره‌ای در ماتریس موقت
                current_loss_matrix[idx] = ano_score_batch.detach().cpu().numpy()
                # ============================================================


                loss.backward()
                optimiser.step()

                loss = loss.detach().cpu().numpy()
                if not is_final_batch:
                    total_loss += loss

            mean_loss = (total_loss * batch_size + loss * cur_batch_size) / nb_nodes

            # === E. ذخیره loss_matrix در انتهای Epoch ===
            loss_matrix_list.append(current_loss_matrix.copy())
            # ==========================================


            if mean_loss < best:
                best = mean_loss
                best_t = epoch
                cnt_wait = 0
                # ذخیره مدل و متغیرهای AD-GCL برای تست
                torch.save(model.state_dict(), '{}.pkl'.format(args.dataset))
                # ذخیره sim و ano_sim نهایی
                torch.save(sim, '{}_sim.pt'.format(args.dataset))
                torch.save(ano_sim, '{}_ano_sim.pt'.format(args.dataset))
            else:
                cnt_wait += 1

            if cnt_wait == args.patience:
                print('Early stopping!', flush=True)
                break

            print('Epoch:{} Loss:{:.8f}'.format(epoch, mean_loss), flush=True)

        # Testing
        print('Loading {}th epoch'.format(best_t), flush=True)
        model.load_state_dict(torch.load('{}.pkl'.format(args.dataset)))
        
        # بارگذاری sim و ano_sim بهترین دوره برای تست
        best_sim = torch.load('{}_sim.pt'.format(args.dataset)).to(device)
        best_ano_sim = torch.load('{}_ano_sim.pt'.format(args.dataset)).to(device)

        # === تولید نماهای نهایی برای Testing ===
        # View 1: هرس (Pruning) - برای GRADATE (adj)
        _, _, _, adj_view1_full_test = neighbor_pruning(
            dgl_graph, node_dist.cpu(), best_sim.cpu(), features.cpu(), 
            degree, args.feat_drop_rate_1, args.feat_drop_rate_1, args.degree_threshold
        )
        adj_view1_full_test = adj_view1_full_test.to(device)
        
        # View 2: تکمیل (Completion) - برای GRADATE (adj_hat)
        _, _, _, _, adj_view2_full_test, _ = neighbor_completion(
            dgl_graph, node_dist.cpu(), best_sim.cpu(), best_ano_sim.cpu(), features.cpu(), 
            degree, args.feat_drop_rate_1, args.edge_mask_rate_1, 
            args.feat_drop_rate_2, args.edge_mask_rate_2, args.degree_threshold, device
        )
        adj_view2_full_test = adj_view2_full_test.to(device)
        # ========================================


        multi_round_ano_score = np.zeros((args.auc_test_rounds, nb_nodes))
        print('Testing AUC!', flush=True)

        with tqdm(total=args.auc_test_rounds) as pbar_test:
            pbar_test.set_description('Testing')
            for round in range(args.auc_test_rounds):
                all_idx = list(range(nb_nodes))
                random.shuffle(all_idx)
                subgraphs = generate_rwr_subgraph(dgl_graph, subgraph_size)
                
                for batch_idx in range(batch_num):
                    optimiser.zero_grad()
                    is_final_batch = (batch_idx == (batch_num - 1))
                    if not is_final_batch:
                        idx = all_idx[batch_idx * batch_size: (batch_idx + 1) * batch_size]
                    else:
                        idx = all_idx[batch_idx * batch_size:]
                    cur_batch_size = len(idx)
                    
                    ba = []
                    ba_hat = []
                    bf = []
                    
                    added_adj_zero_row = torch.zeros((cur_batch_size, 1, subgraph_size)).to(device)
                    added_adj_zero_col = torch.zeros((cur_batch_size, subgraph_size + 1, 1)).to(device)
                    added_adj_zero_col[:, -1, :] = 1.
                    added_feat_zero_row = torch.zeros((cur_batch_size, 1, ft_size)).to(device)
                    
                    for i in idx:
                        # استفاده از نماهای نهایی تولید شده برای تست
                        cur_adj = adj_view1_full_test[:, subgraphs[i], :][:, :, subgraphs[i]]
                        cur_adj_hat = adj_view2_full_test[:, subgraphs[i], :][:, :, subgraphs[i]]
                        cur_feat = features[np.newaxis, subgraphs[i], :]
                        ba.append(cur_adj)
                        ba_hat.append(cur_adj_hat)
                        bf.append(cur_feat)

                    ba = torch.cat(ba)
                    ba = torch.cat((ba, added_adj_zero_row), dim=1)
                    ba = torch.cat((ba, added_adj_zero_col), dim=2)
                    ba_hat = torch.cat(ba_hat)
                    ba_hat = torch.cat((ba_hat, added_adj_zero_row), dim=1)
                    ba_hat = torch.cat((ba_hat, added_adj_zero_col), dim=2)
                    bf = torch.cat(bf)
                    bf = torch.cat((bf[:, :-1, :], added_feat_zero_row, bf[:, -1:, :]), dim=1)


                    with torch.no_grad():
                        test_logits_1, test_logits_2, _, _ = model(bf, ba)
                        test_logits_1_hat, test_logits_2_hat, _, _ = model(bf, ba_hat)
                        test_logits_1 = torch.sigmoid(torch.squeeze(test_logits_1))
                        test_logits_2 = torch.sigmoid(torch.squeeze(test_logits_2))
                        test_logits_1_hat = torch.sigmoid(torch.squeeze(test_logits_1_hat))
                        test_logits_2_hat = torch.sigmoid(torch.squeeze(test_logits_2_hat))


                        ano_score_1 = - (test_logits_1[:cur_batch_size] - torch.mean(test_logits_1[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                        ano_score_1_hat = - (
                                    test_logits_1_hat[:cur_batch_size] - torch.mean(test_logits_1_hat[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_context), dim=1)).cpu().numpy()
                        ano_score_2 = - (test_logits_2[:cur_batch_size] - torch.mean(test_logits_2[cur_batch_size:].view(
                            cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                        ano_score_2_hat = - (
                                    test_logits_2_hat[:cur_batch_size] - torch.mean(test_logits_2_hat[cur_batch_size:].view(
                                cur_batch_size, args.negsamp_ratio_patch), dim=1)).cpu().numpy()
                        
                        ano_score = args.beta * (args.alpha * ano_score_1 + (1 - args.alpha) * ano_score_1_hat)  + \
                                    (1 - args.beta) * (args.alpha * ano_score_2 + (1 - args.alpha) * ano_score_2_hat)

                    multi_round_ano_score[round, idx] = ano_score

                pbar_test.update(1)

            ano_score_final = np.mean(multi_round_ano_score, axis=0) + np.std(multi_round_ano_score, axis=0)
            auc = roc_auc_score(ano_label, ano_score_final)
            all_auc.append(auc)
            print('Testing AUC:{:.4f}'.format(auc), flush=True)


    print('\n==============================')
    print(all_auc)
    print('FINAL TESTING AUC:{:.4f}'.format(np.mean(all_auc)))
    print('==============================')