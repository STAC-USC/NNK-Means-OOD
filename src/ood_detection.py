import time
import torch
import numpy as np
from sklearn.covariance import EmpiricalCovariance
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
import faiss
from ec_nnk_means import NNKMU

def fit_mahalanobis(bank, label_bank, test_bank, device):
    start_time = time.time()
    N, d = bank.size()
    all_classes = list(set(label_bank.tolist()))

    class_mean = torch.zeros(max(all_classes) + 1, d).to(device)
    for c in all_classes:
        class_mean[c] = (bank[label_bank == c].mean(0))

    centered_bank = (bank - class_mean[label_bank]).detach().cpu().numpy()
    precision = EmpiricalCovariance().fit(centered_bank).precision_.astype(
        np.float32)  # .precision_ is the estimated pseudo-inverse matrix.

    class_var = torch.from_numpy(precision).float().to(device)
    fit_time = time.time() - start_time

    start_time = time.time()
    maha_score = []
    for c in all_classes:
        centered_pooled = test_bank - class_mean[c].unsqueeze(0)
        ms = torch.diag(centered_pooled @ class_var @ centered_pooled.t())
        maha_score.append(ms)
    maha_score = torch.stack(maha_score, dim=-1)
    maha_score = maha_score.min(-1)[0]
    maha_score = -maha_score
    predict_time = time.time() - start_time

    return maha_score, fit_time, predict_time

def fit_knn(bank, model, test_bank, k=1):
    start_time = time.time()
    index = faiss.index_factory(model.config.hidden_size, "Flat", faiss.METRIC_INNER_PRODUCT)
    z = bank.detach().clone().cpu().numpy()
    faiss.normalize_L2(z)
    index.add(z)
    fit_time = time.time() - start_time

    start_time = time.time()
    z = test_bank.detach().clone().cpu().numpy()
    faiss.normalize_L2(z)
    scores, _ = index.search(z, 10000)
    scores[scores < 1e-20] = 0   # To avoid underflow for k-avg NN
    knn_distances = -1 * (1 - scores[:, k-1])
    predict_time = time.time() - start_time

    return knn_distances, fit_time, predict_time

def fit_nnkm(bank, args, test_bank):
    start_time = time.time()
    nnk_model = NNKMU(seed=args.seed, num_epochs=args.epochs, metric='error', n_components=args.atoms, ep=args.ep, 
                      weighted=(args.weighted == 0), top_k=args.sparsity, num_warmup=args.warmup, num_cooldown=args.cooldown)
    nnk_model.fit(bank.cpu().numpy(), batch_size=args.batch)
    fit_time = time.time() - start_time

    start_time = time.time()
    nnk_score = nnk_model.predict_score(test_bank.cpu(), bank.cpu())
    nnk_score = -nnk_score
    predict_time = time.time() - start_time

    return nnk_score, fit_time, predict_time, nnk_model
        
def fit_cnnkm(bank, label_bank, args, test_bank):
    start_time = time.time()
    models = []
    all_classes = list(set(label_bank.tolist()))
    for c in all_classes:
        model = NNKMU(seed=args.seed, num_epochs=args.epochs, metric='error', n_components=args.catoms, ep=args.cep, 
                        weighted=(args.weighted == 0), top_k=args.sparsity, num_warmup=args.warmup, num_cooldown=args.cooldown)
        model.fit(bank[label_bank == c].cpu().numpy(), batch_size=args.batch)
        models.append(model)
    fit_time = time.time() - start_time

    start_time = time.time()
    c_nnk_score = []
    for i, c in enumerate(all_classes):
        c_nnk_score.append(models[i].predict_score(test_bank.cpu(), bank[label_bank == c].cpu()))
    c_nnk_score = torch.stack(c_nnk_score, dim=-1)
    c_nnk_score = c_nnk_score.min(-1)[0]
    c_nnk_score = -c_nnk_score
    predict_time = time.time() - start_time

    return c_nnk_score, fit_time, predict_time, models

def fpr_at_95_tpr(preds, labels, pos_label=1):
    """
    from https://github.com/tayden/ood-metrics/blob/main/ood_metrics/metrics.py
    Return the FPR when TPR is at minimum 95%.
        
    preds: array, shape = [n_samples]
           Target normality scores, can either be probability estimates of the positive class, confidence values, or non-thresholded measure of decisions.
           i.e.: an high value means sample predicted "normal", belonging to the positive class
           
    labels: array, shape = [n_samples]
            True binary labels in range {0, 1} or {-1, 1}.

    pos_label: label of the positive class (1 by default)
    """
    fpr, tpr, _ = roc_curve(labels, preds, pos_label=pos_label)

    if all(tpr < 0.95):
        # No threshold allows TPR >= 0.95
        return 0
    elif all(tpr >= 0.95):
        # All thresholds allow TPR >= 0.95, so find lowest possible FPR
        idxs = [i for i, x in enumerate(tpr) if x >= 0.95]
        return min(map(lambda idx: fpr[idx], idxs))
    else:
        # Linear interp between values to get FPR at TPR == 0.95
        return np.interp(0.95, tpr, fpr)

def eval_metrics(y_true, y_score):
    auc = roc_auc_score(y_true, y_score)
    fpr95 = fpr_at_95_tpr(y_score, y_true)
    aucpr = average_precision_score(y_true, y_score)
    return auc, fpr95, aucpr