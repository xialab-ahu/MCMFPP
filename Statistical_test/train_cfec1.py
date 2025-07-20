from torch import optim
from utils import *
from preprocess import *
from models import *
import numpy as np
import torch
import argparse
from tqdm import tqdm
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
    parse.add_argument('-fold', type=float, default=1, help='The fold (1-10) used as validation in 10-fold cross-validation')
    parse.add_argument('-batch_size', type=int, default=256, help='Maximum Batch Size')
    parse.add_argument('-train_direction', type=str, default='ten-fold cross-validation/sub_train_10-fold_9.txt', help='Location of the training set')
    parse.add_argument('-test_direction', type=str, default='ten-fold cross-validation/sub_val_10-fold_9.txt', help='Location of the test set')
    config = parse.parse_args()
    return config

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    time1 = time.time()
    args = get_config()
    for args.fold in [1,2,3,4,5,6,7,8,9,10]:
        args.train_direction = f'ten-fold cross-validation/sub_train_10-fold_{args.fold}.txt'
        args.test_direction=f'ten-fold cross-validation/sub_val_10-fold_{args.fold}.txt'
        set_seed(args.random_seed)
        train_loader_cfec, test_loader=get_cfec_input(args)
        print("The data processing is complete!")
        classifier_CFEC=CFEC(vocab_size=args.vocab_size, position_embedding_dim=args.position_embedding_dim,
                       ESMC_embedding_dim=args.ESMC_embedding_dim, n_filters=args.n_filters,
                       filter_sizes=args.filter_sizes, output_dim=len(args.peptides_name), dropout=args.dropout)
        mlfdl = FocalDiceLoss()
        optimizer_classifier_cfec = optim.Adam(classifier_CFEC.parameters(), lr=args.lr_cfec)
        lr_scheduler_cfec = CosineScheduler(250, base_lr=args.lr_cfec, warmup_steps=20)
        steps = 1
        classifier_CFEC=classifier_CFEC.to(device)
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
                loss_cfec.backward()
                optimizer_classifier_cfec.step()
                total_loss += mlfdl_cfec.item()
                for param_group in optimizer_classifier_cfec.param_groups:
                    param_group['lr'] = lr_scheduler_cfec(steps)
            print(f'Rank {device} |Epoch {epoch + 1}/{args.epochs}, Training loss: {total_loss / len(train_loader_cfec)}')

            classifier_CFEC.eval()
            with torch.no_grad():
                preds = []
                for batch in test_loader:
                    data1, data2, labels = batch
                    data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
                    outputs, _ = classifier_CFEC(data1, data2)
                    pred = torch.sigmoid(outputs)
                    preds.extend(pred.cpu().numpy())
                preds = np.array(preds)
                # np.savetxt('save1/classifier_cfec_preds.csv', preds, delimiter=',')
            steps += 1
        classifier_CFEC.eval()
        torch.save(classifier_CFEC, f'save1/classifier_cfec_10-fold_{args.fold}.pth')
if __name__ == '__main__':
    train()



















