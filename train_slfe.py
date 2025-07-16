from torch import optim
from DBLoss import gain_class_freq_and_train_num, ResampleLoss
from estimate import evaluate
from preprocess import *
from models import *
from utils import *
import numpy as np
import torch
import argparse
from tqdm import tqdm
import time
import os
from torchinfo import summary
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
num_labels = 21
def get_config():
    parse = argparse.ArgumentParser(description='Parameter Management')
    parse.add_argument('-task', type=str, default='MFTP', help='Task type selection: MFTP')
    parse.add_argument('-peptides_name', type=list, default=['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP','BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'], help='Types of therapeutic peptides in the dataset')
    parse.add_argument('-random_seed',type=int, default=20230226 ,help='Random seed (20230226, same as ETFC)')
    parse.add_argument('-criterion', type=str, default='FDL', help='Loss function selection: FDL')
    parse.add_argument('-max_length', type=int, default=52, help='Maximum allowed length for sequence encoding')
    parse.add_argument('-model', type=str, default='SLFE', help='Model selection: SLFE')
    parse.add_argument('-vocab_size', type=int, default=21, help='Vocabulary size')
    parse.add_argument('-position_embedding_dim', type=int, default=128, help='128; if 0, no positional encoding is applied')
    parse.add_argument('-ESMC_embedding_dim', type=int, default=960, help='0 or 960 (0 indicates no ESMC extraction)')
    parse.add_argument('-n_filters', type=int, default=128, help='Number of filters per convolution kernel')
    parse.add_argument('-filter_sizes', type=list, default=[3, 4, 5], help='Sizes of convolution kernels')
    parse.add_argument('-dropout', type=float, default=0.6, help='Dropout rate')
    parse.add_argument('-epochs', type=int, default=200, help='epochs')
    parse.add_argument('-lr_slfe', type=float, default=0.0012, help='Learning rate for SLFE')
    parse.add_argument('-batch_size', type=int, default=256, help='Maximum Batch Size')
    parse.add_argument('-train_direction', type=str, default='MFTP/train.txt', help='Location of the training set')
    parse.add_argument('-test_direction', type=str, default='MFTP/test.txt', help='Location of the test set')
    config = parse.parse_args()
    return config

def train():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args = get_config()
    set_seed(args.random_seed)
    train_loader_slfe, test_loader=get_slfe_input(args)
    print("The data processing is complete!")
    classifier_SLFE=SLFE(vocab_size=args.vocab_size, position_embedding_dim=args.position_embedding_dim,
                   ESMC_embedding_dim=args.ESMC_embedding_dim, n_filters=args.n_filters,
                   filter_sizes=args.filter_sizes, output_dim=len(args.peptides_name), dropout=args.dropout)

    # summary(
    #     classifier_SLFE,
    #     input_size=[(1, 52, 960), (1, 52)],  # 输入形状（含 batch_size）
    #     dtypes=[torch.float32, torch.long],  # 明确指定 dtype
    #     verbose=1,
    # )
    time1=time.time()
    mlfdl = FocalDiceLoss()
    # BCE_loss=nn.BCEWithLogitsLoss()
    # FocalLoss=MultiLabelFocalLoss()
    # DiceLoss=BinaryDiceLoss()
    # ASL=AsymmetricLoss()

    # class_freq, train_num = gain_class_freq_and_train_num()
    # print("训练样本治疗肽的频率", class_freq)
    # print("训练样本的个数", train_num)
    # CBLoss=ClassBalancedLoss(samples_per_cls=class_freq)
    # DBLoss = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
    #                          focal=dict(focal=True, alpha=0.5, gamma=2),
    #                          logit_reg=dict(init_bias=0.05, neg_scale=2.0),
    #                          map_param=dict(alpha=0.1, beta=10.0, gamma=0.9),
    #                          class_freq=class_freq, train_num=train_num)


    optimizer_classifier_slfe = optim.Adam(classifier_SLFE.parameters(), lr=args.lr_slfe)
    lr_scheduler_slfe = CosineScheduler(250, base_lr=args.lr_slfe, warmup_steps=20)
    steps = 1
    # classifier_SLFE=torch.load('save/classifier_slfe.pth')
    classifier_SLFE=classifier_SLFE.to(device)
    for epoch in tqdm(range(args.epochs)):
        classifier_SLFE.train()
        total_loss = 0
        for batch in train_loader_slfe:
            data1, data2, labels = batch
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            optimizer_classifier_slfe.zero_grad()
            outputs, _ = classifier_SLFE(data1, data2)
            loss_slfe = mlfdl(outputs, labels.float())
            # loss_slfe = BCE_loss(outputs, labels.float())
            # loss_slfe = FocalLoss(outputs, labels.float())
            # loss_slfe = DBLoss(outputs, labels.float())
            # loss_slfe = DiceLoss(outputs, labels.float())
            # loss_slfe = ASL(outputs, labels.float())
            # loss_slfe = CBLoss(outputs, labels.float())
            loss_slfe.backward()
            optimizer_classifier_slfe.step()
            total_loss += loss_slfe.item()
            for param_group in optimizer_classifier_slfe.param_groups:
                param_group['lr'] = lr_scheduler_slfe(steps)
        print(f'Rank {device} |Epoch {epoch + 1}/{args.epochs}, Training loss: {total_loss / len(train_loader_slfe)}')
        classifier_SLFE.eval()
        with torch.no_grad():
            preds = []
            reals = []
            for batch in test_loader:
                data1, data2, labels = batch
                data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
                outputs, _ = classifier_SLFE(data1, data2)
                pred = torch.sigmoid(outputs)
                preds.extend(pred.cpu().numpy())
            # np.savetxt('save/classifier_slfe_preds.csv', preds, delimiter=',')
        steps += 1
    time2 = time.time()
    print("训练时间为：", time2-time1,"s")
    classifier_SLFE.eval()
    with torch.no_grad():
        preds = []
        reals = []
        for batch in test_loader:
            data1, data2, labels = batch
            data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
            outputs, _ = classifier_SLFE(data1, data2)
            pred = torch.sigmoid(outputs)
            preds.extend(pred.cpu().numpy())
            reals.extend(labels.cpu().numpy())
        preds = np.array(preds)
        reals = np.array(reals)
        score = evaluate(preds, reals)
        print("=====SLFE=====Test=====Performance Evaluation======")
        print(f'aiming: {score[0]:.3f}')
        print(f'coverage: {score[1]:.3f}')
        print(f'accuracy: {score[2]:.3f}')
        print(f'absolute_true: {score[3]:.3f}')
        print(f'absolute_false: {score[4]:.3f}\n')
    # classifier_SLFE.eval()
    time3 = time.time()
    print("推理时间：", (time3-time2)/1969,"s")
    torch.save(classifier_SLFE, f'save/classifier_slfe.pth')

if __name__ == '__main__':
    train()

















