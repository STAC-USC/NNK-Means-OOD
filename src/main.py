import os, argparse, sys
import torch
import random
import numpy as np
from transformers import AutoModel, AutoTokenizer

from data import load_data_20ng, preprocess_data, extract_embeddings
from ood_detection import fit_mahalanobis, fit_knn, fit_nnkm, fit_cnnkm, eval_metrics

# Constants
USE_VAL = True
MODEL_NAME = f'sentence-transformers/all-distilroberta-v1'
TOKENIZER = f'sentence-transformers/all-distilroberta-v1'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--ep', type=float, default=0.03, help='lambda for the NNKMU Entropy Constraint')
    parser.add_argument('--cep', type=float, default=0.03, help='lambda for the C-NNKM Entropy Constraint')
    parser.add_argument('--weighted', type=int, default=1, help='0 = Weighted NNKMU')
    parser.add_argument('--atoms', type=int, default=350, help='Number of atoms for the dictionary')
    parser.add_argument('--catoms', type=int, default=350, help='Number of atoms for the class dictionary')
    parser.add_argument('--sparsity', type=int, default=20, help='Sparsity for the dictionary')
    parser.add_argument('--known', type=float, default=0.25, help='Ratio of Known Classes to Unknown')
    parser.add_argument('--epochs', type=int, default=8, help='Number of epochs for training')
    parser.add_argument('--warmup', type=int, default=0, help='Number of warm-up epochs before entropy-constraining')
    parser.add_argument('--cooldown', type=int, default=2, help='Number of epochs at end without entropy-constraint')
    parser.add_argument('--batch', type=int, default=1024, help='Batch Size')
    parser.add_argument('--seed', type=int, default=-1, help='Random Seed')

    args = parser.parse_args()

    ep = args.ep
    weighted = (args.weighted == 0)
    atoms = args.atoms
    sparsity = args.sparsity
    epochs = args.epochs
    warmup = args.warmup
    cooldown = args.cooldown
    batch_size = args.batch
    known_ratio = args.known
    seed = args.seed if args.seed > 0 else random.randint(1, 500)
    c_atoms = args.catoms
    cep = args.cep

    return args, ep, weighted, atoms, sparsity, epochs, warmup, cooldown, batch_size, known_ratio, seed, c_atoms, cep


def print_results(fit_times, predict_times, aucs, fpr95s, aucprs, methods):
    for i, method in enumerate(methods):
        print(method)
        print("AUC: ", aucs[i])
        print("FPR95: ", fpr95s[i])
        print("AUCPR: ", aucprs[i])
        print("Fit Time: ", fit_times[i])
        print("Predict Time: ", predict_times[i])

if __name__ == '__main__':
    args, ep, weighted, atoms, sparsity, epochs, warmup, cooldown, batch_size, known_ratio, seed, c_atoms, cep = parse_args()
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print (f"ep: {ep}, weighted: {weighted}, atoms: {atoms}, sparsity: {sparsity}, epochs: {epochs}, warmup: {warmup}, cooldown: {cooldown}, batch_size: {batch_size}, known_ratio: {known_ratio}, seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = AutoModel.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    processed_dataset, known_labels = load_data_20ng(known_ratio=known_ratio) # replace with any dataset from data.py
    train_dataloader, val_dataloader, test_dataloader = preprocess_data(processed_dataset, known_labels, tokenizer)

    if USE_VAL:
        bank, label_bank, test_bank, ood_labels, test_labels = extract_embeddings(train_dataloader, val_dataloader, device, model, known_labels)    
    else:
        bank, label_bank, test_bank, ood_labels, test_labels = extract_embeddings(train_dataloader, test_dataloader, device, model, known_labels)

    fit_times = []
    predict_times = []
    aucs = []
    fpr95s = []
    aucprs = []
    methods = ['Mahalanobis', 'kNN', 'NNK-Means', 'C-NNK-Means']

    print("Fitting Mahalanobis")
    maha_score, fit_time, predict_time = fit_mahalanobis(bank, label_bank, test_bank, device)
    auc, fpr95, aucpr = eval_metrics(ood_labels.cpu().numpy(), maha_score.cpu().numpy())
    aucs.append(auc)
    fpr95s.append(fpr95)
    aucprs.append(aucpr)
    fit_times.append(fit_time)
    predict_times.append(predict_time)

    print("Fitting kNN")
    knn_distances, fit_time, predict_time = fit_knn(bank, model, test_bank)
    auc, fpr95, aucpr = eval_metrics(ood_labels.cpu().numpy(), knn_distances)
    aucs.append(auc)
    fpr95s.append(fpr95)
    aucprs.append(aucpr)
    fit_times.append(fit_time)
    predict_times.append(predict_time)

    print("Fitting NNK-Means")
    nnk_score, fit_time, predict_time, nnk_model = fit_nnkm(bank, args, test_bank)
    auc, fpr95, aucpr = eval_metrics(ood_labels.cpu().numpy(), nnk_score)
    aucs.append(auc)
    fpr95s.append(fpr95)
    aucprs.append(aucpr)
    fit_times.append(fit_time)
    predict_times.append(predict_time)

    print("Fitting Class-wise NNK-Means")
    c_nnk_score, fit_time, predict_time, c_nnk_models = fit_cnnkm(bank, label_bank, args, test_bank)
    auc, fpr95, aucpr = eval_metrics(ood_labels.cpu().numpy(), c_nnk_score)
    aucs.append(auc)
    fpr95s.append(fpr95)
    aucprs.append(aucpr)
    fit_times.append(fit_time)
    predict_times.append(predict_time)

    print_results(fit_times, predict_times, aucs, fpr95s, aucprs, methods)