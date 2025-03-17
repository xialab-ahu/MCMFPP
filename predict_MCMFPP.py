import argparse
import math

import torch
from torch import nn

from estimate import *
from preprocess import *

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class MCMFPP():
    def __init__(self, head_label, tail_label, weight1=0.55, weight2=0):
        super(MCMFPP, self).__init__()
        self.head_label = head_label
        self.tail_label = tail_label
        self.weight1 = weight1
        self.weight2 = weight2

    def predict(self, classifier_SLFE,classifier_CFEC,test_loader):
        classifier_SLFE_preds,_ = self.generate_predictions(classifier_SLFE,test_loader)
        classifier_CFEC_preds,reals = self.generate_predictions(classifier_CFEC,test_loader)
        preds = np.copy(classifier_SLFE_preds)
        for i in self.head_label:
            preds[:, i] = (1-self.weight1) * classifier_SLFE_preds[:, i] + self.weight1 * classifier_CFEC_preds[:, i]
        for i in self.tail_label:
            preds[:, i] = (1-self.weight2) * classifier_SLFE_preds[:, i] + self.weight2 * classifier_CFEC_preds[:, i]
        return preds,reals

    def generate_predictions(self,classifier, test_loader):
        preds = []
        reals = []
        classifier.eval()
        with torch.no_grad():
            for batch in test_loader:
                data1, data2, labels = batch
                data1, data2, labels = data1.to(device), data2.to(device), labels.to(device)
                outputs, _ = classifier(data1, data2)
                pred = torch.sigmoid(outputs)
                preds.extend(pred.cpu().numpy())
                reals.extend(labels.cpu().numpy())
            preds = np.array(preds)
            reals = np.array(reals)
            return preds, reals

class SeqCNN(nn.Module):
    def __init__(self, vocab_size:int,position_embedding_dim:int,ESMC_embedding_dim:int,n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float):
        super(SeqCNN, self).__init__()
        self.position_embedding_dim=position_embedding_dim
        self.ESMC_embedding_dim=ESMC_embedding_dim
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=position_embedding_dim+ESMC_embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        if position_embedding_dim!=0:
            self.embedding = nn.Embedding(vocab_size, position_embedding_dim)
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters*5),
                                nn.Mish(),  #nn.Mish是一种非单调的激活函数Mish(x)=x*tanh(ln(1+e^x))
                                nn.Dropout(),
            nn.Linear(len(filter_sizes) * n_filters * 5, len(filter_sizes) * n_filters),
            nn.Mish(),  # nn.Mish是一种非单调的激活函数Mish(x)=x*tanh(ln(1+e^x))
            # nn.Dropout(),
            nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2),
            nn.Mish(),

        )
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()
        self.fc1=nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)

    def forward(self, data1,data2):
        if  self.position_embedding_dim!=0 and self.ESMC_embedding_dim!=0:
            # 确保data1是float32类型
            data1 = data1.to(dtype=torch.float32)
            # 应用嵌入层到data2，假设data2是索引形式的输入
            embedded2 = self.embedding(data2)
            # 调整data1和embedded2的维度以匹配，这里假设data1已经是嵌入后的数据
            # 连接embedded1和embedded2
            embedded = torch.cat((data1, embedded2), dim=-1)
            # 调整embedded2的维度以匹配embedded1
            embedded = embedded.permute(0, 2, 1)
        elif self.position_embedding_dim!=0:
            embedded = self.embedding(data2.long())
            embedded = embedded.permute(0, 2, 1)
        else:
            embedded = data1.to(dtype=torch.float32)
            embedded = embedded.permute(0, 2, 1)
        # 多分支卷积
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        # 多分支最大池化
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        # 多分支线性展开
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        # 将各分支连接在一起
        cat = self.dropout(torch.cat(flatten, dim=1))  # （256， 9216）
        # 输入全连接层，进行回归输出
        cat1 = self.fc(cat)
        cat2 = self.fc1(cat1)
        return cat2, cat1

def get_config():
    # 参数管理
    parse = argparse.ArgumentParser(description='参数管理')
    #分类任务
    parse.add_argument('-task', type=str, default='MFTP', help='任务类型选择：MFBP or MFTP')
    parse.add_argument('-peptides_name', type=list, default=['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP','BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'], help='数据集中治疗肽的种类')
    #复现参数
    parse.add_argument('-random_seed',type=int, default=20230226 ,help='随机种子 20230226 与ETFC保持一致')
    #损失函数
    parse.add_argument('-criterion', type=str, default='FDL', help='损失函数选择：FDL')
    #模型参数
    parse.add_argument('-max_length', type=int, default=52, help='序列词编码最大允许长度')
    parse.add_argument('-model', type=str, default='SeqCNN', help='选择模型:SeqCNN')
    parse.add_argument('-vocab_size', type=int, default=21, help='词汇表大小，即词汇种类的总数')
    parse.add_argument('-position_embedding_dim', type=int, default=128, help='位置编码词向量的嵌入维度128，如果等于0，不进行位置编码')
    parse.add_argument('-ESMC_embedding_dim', type=int, default=960, help='0 or 960,0代表不使用ESMC提取')
    parse.add_argument('-n_filters', type=int, default=128, help='每个卷积核的数量，即卷积后的表征样本词特征的通道数')
    parse.add_argument('-filter_sizes', type=list, default=[3, 4, 5], help='卷积核的大小，列表中的每个元素表示一个卷积核的大小')
    parse.add_argument('-dropout', type=float, default=0.6, help='丢弃率，防止过拟合')
    #训练参数
    parse.add_argument('-epochs', type=int, default=200, help='训练回合:300')
    parse.add_argument('-N', type=int, default=2, help='增强倍数:0-5')
    parse.add_argument('-lr_slfe', type=float, default=0.0012, help='slfe学习率')
    parse.add_argument('-lr_cfec', type=float, default=0.0007, help='cfec学习率')
    parse.add_argument('-threshold', type=float, default=0.6, help='阈值，用于模型评估')
    parse.add_argument('-batch_size', type=int, default=256, help='批次大小')
    #MCMFPP模型融合参数
    parse.add_argument('-W1', type=float, default=0.55, help='CFEC子分类预测头类的权重')
    parse.add_argument('-W2', type=float, default=0, help='CFEC子分类预测尾类的权重')
    #数据集加载路径
    parse.add_argument('-test_direction', type=str, default='MFTP/test.txt', help='测试集的存储位置')
    config = parse.parse_args()
    return config
args = get_config()
head_label = [1, 2, 4, 6, 8, 9, 11, 12, 13, 15, 16, 17, 20]
tail_label = [0, 3, 5, 7, 10, 14, 19,18]
mcmfpp=MCMFPP(head_label, tail_label,weight1=args.W1, weight2=args.W2)
classifier_SLFE=torch.load('save/model_SLFE.pth')
classifier_CFEC=torch.load('save/model_CFEC.pth')
test_loader=get_slfe_input(args)
preds,reals=mcmfpp.predict(classifier_SLFE,classifier_CFEC,test_loader)
score = evaluate(preds, reals, threshold=args.threshold)
print("=====MFTP融合分类器=====测试集=====性能评估======")
print(f'aiming: {score[0]:.3f}')
print(f'coverage: {score[1]:.3f}')
print(f'accuracy: {score[2]:.3f}')
print(f'absolute_true: {score[3]:.3f}')
print(f'absolute_false: {score[4]:.3f}\n')
