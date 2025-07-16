from torch import optim
from utils import *
from preprocess import *
from models import *
import numpy as np
import torch
import argparse
from tqdm import tqdm
from torchinfo import summary
import estimate
import time
import os
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
num_labels = 21
def get_config():
    parse = argparse.ArgumentParser(description='Parameter Management')
    parse.add_argument('-task', type=str, default='MFTP', help='Task type selection: MFTP')
    parse.add_argument('-peptides_name', type=list, default=['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP','BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'], help='Types of therapeutic peptides in the dataset')
    parse.add_argument('-random_seed',type=int, default=20230226 ,help='Random seed (20230226, same as ETFC)')
    parse.add_argument('-criterion', type=str, default='FDL', help='Loss function selection: MLFDL')
    parse.add_argument('-max_length', type=int, default=52, help='Maximum allowed length for sequence encoding')
    parse.add_argument('-model', type=str, default='CFEC', help='Model selection: CFEC')
    parse.add_argument('-vocab_size', type=int, default=21, help='Vocabulary size')
    parse.add_argument('-position_embedding_dim', type=int, default=128, help='128; if 0, no positional encoding is applied')
    parse.add_argument('-ESMC_embedding_dim', type=int, default=960, help='0 or 960 (0 indicates no ESMC extraction)')
    parse.add_argument('-n_filters', type=int, default=128, help='Number of filters per convolution kernel')
    parse.add_argument('-filter_sizes', type=list, default=[3, 4, 5], help='Sizes of convolution kernels')
    parse.add_argument('-dropout', type=float, default=0.6, help='Dropout rate')
    parse.add_argument('-epochs', type=int, default=200, help='epochs')
    parse.add_argument('-N', type=int, default=2, help='Augmentation multiplier (0-5)')
    parse.add_argument('-lr_cfec', type=float, default=0.0007, help='Learning rate for CFEC')
    parse.add_argument('-batch_size', type=int, default=256, help='Maximum Batch Size')
    parse.add_argument('-train_direction', type=str, default='MFTP/train.txt', help='Location of the training set')
    parse.add_argument('-test_direction', type=str, default='MFTP/test.txt', help='Location of the test set')
    config = parse.parse_args()
    return config

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time1 = time.time()
    args = get_config()
    set_seed(args.random_seed)
    train_loader_cfec, test_loader=get_cfec_input(args)
    print("The data processing is complete!")
    classifier_CFEC=CFEC(vocab_size=args.vocab_size, position_embedding_dim=args.position_embedding_dim,
                   ESMC_embedding_dim=args.ESMC_embedding_dim, n_filters=args.n_filters,
                   filter_sizes=args.filter_sizes, output_dim=len(args.peptides_name), dropout=args.dropout)

    # summary(
    #     classifier_CFEC,
    #     input_size=[(1, 52, 960), (1, 52)],  # 输入形状（含 batch_size）
    #     dtypes=[torch.float32, torch.long],  # 明确指定 dtype
    #     verbose=1,
    # )
    classifier_CFEC=classifier_CFEC.to(device)
    mlfdl = FocalDiceLoss()
    optimizer_classifier_cfec = optim.Adam(classifier_CFEC.parameters(), lr=args.lr_cfec)
    lr_scheduler_cfec = CosineScheduler(250, base_lr=args.lr_cfec, warmup_steps=20)
    steps = 1
    time1 = time.time()
    for epoch in tqdm(range(args.epochs)):
        classifier_CFEC.train()
        total_loss = 0
        for batch in train_loader_cfec:
            data1, data2, labels = batch
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            optimizer_classifier_cfec.zero_grad()
            outputs, features = classifier_CFEC(data1, data2)
            mlfdl_cfec = mlfdl(outputs, labels.float())
            mulsupcon_cfec = SCL.mulsupcon(features, labels, temperature=0.5)
            loss_cfec = mlfdl_cfec + 0.1 * mulsupcon_cfec
            # loss_cfec = mlfdl_cfec
            loss_cfec.backward()
            optimizer_classifier_cfec.step()
            total_loss += mlfdl_cfec.item()
            for param_group in optimizer_classifier_cfec.param_groups:
                param_group['lr'] = lr_scheduler_cfec(steps)
        print(f'Rank {device} |Epoch {epoch + 1}/{args.epochs}, Training loss: {total_loss / len(train_loader_cfec)}')

        classifier_CFEC.eval()
        with torch.no_grad():
            preds = []
            reals = []
            for batch in test_loader:
                data1, data2, labels = batch
                data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
                outputs, _ = classifier_CFEC(data1, data2)
                pred = torch.sigmoid(outputs)
                preds.extend(pred.cpu().numpy())
                # reals.extend(labels.cpu().numpy())
            # preds = np.array(preds)
            # np.savetxt('save/classifier_cfec_preds.csv', preds, delimiter=',')
        steps += 1
    # classifier_CFEC = torch.load('save/classifier_cfec_BA.pth')
    # classifier_CFEC = classifier_CFEC.to(device)
    time2 = time.time()
    print("训练时间为：", time2-time1,"s")
    classifier_CFEC.eval()
    preds = []
    reals = []
    for batch in test_loader:
        data1, data2, labels = batch
        data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
        outputs, _ = classifier_CFEC(data1, data2)
        pred = torch.sigmoid(outputs)
        preds.extend(pred.detach().cpu().numpy())
        reals.extend(labels.cpu().numpy())
    preds = np.array(preds)
    reals = np.array(reals)
    score = estimate.evaluate(preds, reals)
    print("=====CFEC=====Test=====Performance Evaluation======")
    print(f'aiming: {score[0]:.3f}')
    print(f'coverage: {score[1]:.3f}')
    print(f'accuracy: {score[2]:.3f}')
    print(f'absolute_true: {score[3]:.3f}')
    print(f'absolute_false: {score[4]:.3f}\n')
    time3 = time.time()
    print("推理时间：", (time3-time2)/1969,"s")
    torch.save(classifier_CFEC, 'save/classifier_cfec.pth')
if __name__ == '__main__':
    train()



















