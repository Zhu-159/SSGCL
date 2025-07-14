import numpy as np
import torch
import torch.nn as nn
from numpy import *
import random
from sklearn import metrics
from torch.optim import Adam
from torch_geometric.data import DataLoader


def sample_with_j(k, n, j):
    if n >= k:
        raise ValueError("n must be less than k.")
    if j < 0 or j > k:
        raise ValueError("j must be in the range 0 to k.")

    numbers = list(range(k))

    if j not in numbers:
        raise ValueError("j must be in the range 0 to k.")

    sample = [j]

    remaining = [num for num in numbers if num != j]
    sample.extend(random.sample(remaining, n - 1))
    return sample


# Frequency Contrastive Regularization
class FCR(nn.Module):
    def __init__(self, ablation=False):
        super(FCR, self).__init__()
        self.l1 = nn.L1Loss()
        self.multi_n_num = 2

    def fft2_wrapper(self, x):
        if hasattr(torch, 'fft') and callable(getattr(torch, 'fft')):
            x_complex = torch.stack([x, torch.zeros_like(x)], dim=-1)
            return torch.fft(x_complex, 2)
        elif hasattr(torch, 'fft') and hasattr(torch.fft, 'fft2'):
            return torch.fft.fft2(x)
        else:
            print("Warning: No FFT implementation found, using original features as substitute")
            return x

    def normalize_fft(self, fft_result, x):
        if fft_result.dim() > x.dim():
            fft_result = torch.sqrt(fft_result[..., 0] ** 2 + fft_result[..., 1] ** 2)

        eps = 1e-8
        norm = torch.norm(fft_result, p=2, dim=1, keepdim=True)
        fft_norm = fft_result / (norm + eps)

        scale = 0.01
        fft_norm = fft_norm * scale

        return fft_norm

    def forward(self, a, p, n):
        a_fft = self.fft2_wrapper(a)
        p_fft = self.fft2_wrapper(p)
        n_fft = self.fft2_wrapper(n)

        a_fft_norm = self.normalize_fft(a_fft, a)
        p_fft_norm = self.normalize_fft(p_fft, p)
        n_fft_norm = self.normalize_fft(n_fft, n)

        contrastive = 0
        for i in range(a_fft_norm.shape[0]):
            d_ap = self.l1(a_fft_norm[i], p_fft_norm[i])
            for j in sample_with_j(a_fft_norm.shape[0], self.multi_n_num, i):
                d_an = self.l1(a_fft_norm[i], n_fft_norm[j])
                contrastive += (d_ap / (d_an + 1e-7))
        contrastive = contrastive / (self.multi_n_num * a_fft_norm.shape[0])
        return contrastive


def train_epochs(train_dataset, test_dataset, model, args):
    num_workers = 2
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True,
                              num_workers=num_workers)

    test_size = 1024
    test_loader = DataLoader(test_dataset, test_size, shuffle=False,
                             num_workers=num_workers)

    model.to(args.device).reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), args.lr)

    fcr = FCR().to(args.device)

    start_epoch = 1
    pbar = range(start_epoch, args.epochs + start_epoch)
    best_epoch, best_auc, best_aupr = 0, 0, 0
    best_precision, best_recall, best_f1 = 0, 0, 0

    for epoch in pbar:
        train_loss = train(model, optimizer, train_loader, args.device, fcr, args.lambda_reg)
        if epoch % args.valid_interval == 0:
            metrics_dict = evaluate_metric(model, test_loader, args.device, epoch)

            roc_auc = metrics_dict['auroc']
            aupr = metrics_dict['aupr']
            precision = metrics_dict['precision']
            recall = metrics_dict['recall']
            f1 = metrics_dict['f1']

            print("epoch {}".format(epoch),
                  "train_loss {0:.4f}".format(train_loss),
                  "AUROC {0:.4f}".format(roc_auc),
                  "AUPR {0:.4f}".format(aupr),
                  "Precision {0:.4f}".format(precision),
                  "Recall {0:.4f}".format(recall),
                  "F1 {0:.4f}".format(f1))

            if roc_auc > best_auc:
                best_epoch, best_auc, best_aupr = epoch, roc_auc, aupr
                best_precision, best_recall, best_f1 = precision, recall, f1

    print("best_epoch {}".format(best_epoch),
          "best_AUROC {0:.4f}".format(best_auc),
          "best_AUPR {0:.4f}".format(best_aupr),
          "best_Precision {0:.4f}".format(best_precision),
          "best_Recall {0:.4f}".format(best_recall),
          "best_F1 {0:.4f}".format(best_f1))

    return {
        'auroc': best_auc,
        'aupr': best_aupr,
        'precision': best_precision,
        'recall': best_recall,
        'f1': best_f1
    }


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)


def train(model, optimizer, loader, device, fcr, lambda_reg):
    model.train()
    total_loss = 0
    pbar = loader

    for data in pbar:
        optimizer.zero_grad()
        true_label = data.to(device)
        predict = model(true_label)
        loss_function = torch.nn.BCEWithLogitsLoss()
        main_loss = loss_function(predict, true_label.y.view(-1))

        batch_size = predict.size(0)

        if batch_size > 1:
            try:
                x = true_label.x

                graph_feats = []
                for i in range(batch_size):
                    mask = true_label.batch == i
                    if mask.sum() > 0:
                        graph_feat = x[mask].mean(dim=0)
                        graph_feats.append(graph_feat)

                if len(graph_feats) > 1:
                    graph_feats = torch.stack(graph_feats)

                    graph_feats = graph_feats.unsqueeze(-1)

                    p_feats = graph_feats[torch.randperm(graph_feats.size(0))]
                    n_feats = graph_feats[torch.randperm(graph_feats.size(0))]

                    fcr_loss = fcr(graph_feats, p_feats, n_feats)
                    loss = main_loss + lambda_reg * fcr_loss
                else:
                    loss = main_loss
            except Exception as e:
                print(f"FCR calculation error: {e}")
                loss = main_loss
        else:
            loss = main_loss

        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        optimizer.step()
        torch.cuda.empty_cache()

    return total_loss / len(loader.dataset)


def evaluate_metric(model, loader, device, epoch):
    model.eval()
    all_y_true = []
    all_y_score = []

    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data)

            y_true = data.y.view(-1).cpu().numpy()
            y_score = out.cpu().numpy()

            all_y_true.append(y_true)
            all_y_score.append(y_score)

            torch.cuda.empty_cache()

    all_y_true = np.concatenate(all_y_true)
    all_y_score = np.concatenate(all_y_score)

    fpr, tpr, _ = metrics.roc_curve(all_y_true, all_y_score)
    roc_auc = metrics.auc(fpr, tpr)

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(all_y_true, all_y_score)
    aupr = metrics.auc(recall_curve, precision_curve)

    y_pred = (all_y_score >= 0.5).astype(int)

    precision = metrics.precision_score(all_y_true, y_pred, zero_division=0)
    recall = metrics.recall_score(all_y_true, y_pred, zero_division=0)
    f1 = metrics.f1_score(all_y_true, y_pred, zero_division=0)

    return {
        'auroc': roc_auc,
        'aupr': aupr,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }