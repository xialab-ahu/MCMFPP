import argparse
import os
import random
import numpy as np
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amino_acids = "XACDEFGHIKLMNPQRSTVWY"

def get_config():
    parse = argparse.ArgumentParser(description='Parameter Management')
    parse.add_argument('-task', type=str, default='MFTP', help='Task type selection: MFTP')
    parse.add_argument('-peptides_name', type=list, default=['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
        'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'], help='Types of therapeutic peptides in the dataset')
    parse.add_argument('-random_seed', type=int, default=20230226, help='Random seed')
    parse.add_argument('-test_direction', type=str, default='../../MFTP/test.txt', help='Location of the test set')
    config = parse.parse_args()
    return config

def set_seed(seed):  # Fix the random seed
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


test_sample, test_label = loadtxt(args.test_direction)

# 转成 list of tuples，方便 torch Dataset 操作
data = list(zip(test_sample, test_label))

# 构造 5 个 80% 的子集
for i in range(5):
    set_seed(args.random_seed+i) # 控制每次随机性
    sub_size = int(0.8 * len(data))
    remainder = len(data) - sub_size

    subset, _ = torch.utils.data.random_split(data, [sub_size, remainder])
    subset = list(subset)  # 转回列表

    out_path = f'sub_test_{i + 1}.txt'
    with open(out_path, "w") as f_out:
        for seq, label in subset:
            label_str = "".join(map(str, label))
            f_out.write(f">{label_str}\n{seq}\n")

    print(f"Saved test subset {i + 1} to {out_path} with {len(subset)} samples")
