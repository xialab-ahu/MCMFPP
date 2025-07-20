import argparse
import os
import random
import numpy as np
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
def get_config():
    parse = argparse.ArgumentParser(description='Parameter Management')
    parse.add_argument('-task', type=str, default='MFTP', help='Task type selection: MFTP')
    parse.add_argument('-peptides_name', type=list, default=['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP','BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'], help='Types of therapeutic peptides in the dataset')
    parse.add_argument('-random_seed',type=int, default=20230226 ,help='Random seed')
    parse.add_argument('-train_direction', type=str, default='../../MFTP/train.txt', help='Location of the training set')
    config = parse.parse_args()
    return config
def set_seed(seed):#Fix the random seed
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def loadtxt(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    sequences = []
    labels = []
    for i in range(0, len(lines), 2):
        label = lines[i][1:].strip()
        sequence = lines[i + 1].strip().upper()
        if any(aa not in amino_acids for aa in sequence):
            continue
        labels.append(list(map(int, label)))
        sequences.append(sequence)
    return sequences, labels
args = get_config()
set_seed(args.random_seed)
train_sample, train_label = loadtxt(args.train_direction)
X = np.array(train_sample)
y = np.array(train_label)
mskf = MultilabelStratifiedKFold(n_splits=10, shuffle=True, random_state=args.random_seed)

for fold, (train_idx, val_idx) in enumerate(mskf.split(X, y)):
    print("=======================")
    print(f"Fold {fold + 1}")
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    print("Label counts in training set (y_train):", y_train.sum(axis=0))
    print("Label counts in validation set (y_val):", y_val.sum(axis=0))
    print("=======================")
    sub_train_dir=f'sub_train_10-fold_{fold+1}.txt'
    # Save to the training set
    with open(sub_train_dir, "w") as f_train:
        for seq, label in zip(X_train, y_train):
            label_str = "".join(map(str, label))
            f_train.write(f">{label_str}\n{seq}\n")
    sub_val_dir = f'sub_val_10-fold_{fold+1}.txt'
    # Save to the val set
    with open(sub_val_dir, "w") as f_val:
        for seq, label in zip(X_val, y_val):
            label_str = "".join(map(str, label))
            f_val.write(f">{label_str}\n{seq}\n")