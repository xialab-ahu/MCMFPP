import math
import random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
device='cuda'
class MCMFPP():
    def __init__(self, head_label, tail_label, weight1=0.55, weight2=0.4):
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


class SLFE(nn.Module):
    def __init__(self, vocab_size: int, position_embedding_dim: int, ESMC_embedding_dim: int, n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float):
        super(SLFE, self).__init__()
        self.position_embedding_dim = position_embedding_dim
        self.ESMC_embedding_dim = ESMC_embedding_dim
        self.text_cnn = TextCNN(vocab_size, position_embedding_dim, ESMC_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.mlp = MLP(vocab_size, position_embedding_dim, ESMC_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.fc = FC(vocab_size, position_embedding_dim, ESMC_embedding_dim, n_filters, filter_sizes, output_dim, dropout)

    def forward(self, data1, data2):
        cat = self.text_cnn(data1, data2)
        Representation = self.mlp(cat)
        logits = self.fc(Representation)
        return logits, Representation

class CFEC(nn.Module):
    def __init__(self, vocab_size: int, position_embedding_dim: int, ESMC_embedding_dim: int, n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float):
        super(CFEC, self).__init__()
        self.position_embedding_dim = position_embedding_dim
        self.ESMC_embedding_dim = ESMC_embedding_dim
        self.text_cnn = TextCNN(vocab_size, position_embedding_dim, ESMC_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.mlp = MLP(vocab_size, position_embedding_dim, ESMC_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.fc = FC(vocab_size, position_embedding_dim, ESMC_embedding_dim, n_filters, filter_sizes, output_dim, dropout)
        self.scl=SCL(192, 128)

    def forward(self, data1, data2):
        cat = self.text_cnn(data1, data2)
        Representation = self.mlp(cat)
        logits = self.fc(Representation)
        Feature = self.scl(Representation)
        return logits, Feature

    @classmethod
    def augment_peptide_dataset_with_reversals(cls,data, label, N):
        augmented_peptides = []
        augmented_labels = []
        i = 0
        for peptide, peptide_label in zip(data, label):
            augmented_peptides.append(peptide)
            augmented_labels.append(peptide_label[:])
            positions_to_check = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            should_augment = any(peptide_label[pos] == 1 for pos in positions_to_check)
            if should_augment == True and np.sum(peptide_label) == 1:
                i = i + 1
                augmented_peptides.append(peptide[::-1])
                augmented_labels.append(peptide_label[:])
                for _ in range(N):
                    mask_position = random.randint(0, len(peptide) - 1)
                    masked_peptide = peptide[:mask_position] + 'X' + peptide[mask_position + 1:]
                    augmented_peptides.append(masked_peptide)
                    augmented_labels.append(peptide_label[:])
                    reversed_masked_peptide = masked_peptide[::-1]
                    augmented_peptides.append(reversed_masked_peptide)
                    augmented_labels.append(peptide_label[:])
        augmented_data = list(zip(augmented_peptides, augmented_labels))
        random.shuffle(augmented_data)
        augmented_peptides, augmented_labels = zip(*augmented_data)
        return list(augmented_peptides), list(augmented_labels)

class SCL(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SCL, self).__init__()
        self.Projection=ProjectionHead(in_dim,out_dim)

    def forward(self,Representation):
        Feature=self.Projection(Representation)
        return Feature

    @classmethod
    def mulsupcon(cls,features, labels, temperature=0.5, eps=1e-9):
        features = F.normalize(features, p=2, dim=1)
        labels = labels.float()
        label_intersection = torch.matmul(labels, labels.T)  # (batch_size, batch_size)
        label_union = (labels.unsqueeze(1) + labels.unsqueeze(0)).clamp(0, 1).sum(dim=-1)
        label_weight = label_intersection / (label_union + eps)
        sim = torch.matmul(features, features.T) / temperature
        exp_sim = torch.exp(sim)
        sim_sum = exp_sim.sum(dim=1) - exp_sim.diag()
        pos_sim = (exp_sim * label_weight).sum(dim=1) - exp_sim.diag()
        loss = -torch.log((pos_sim + eps) / (sim_sum + eps)).mean()
        return loss


class TextCNN(nn.Module):
    def __init__(self, vocab_size:int,position_embedding_dim:int,ESMC_embedding_dim:int,n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float):
        super(TextCNN, self).__init__()
        self.position_embedding_dim=position_embedding_dim
        self.ESMC_embedding_dim=ESMC_embedding_dim
        self.convs = nn.ModuleList([nn.Conv1d(in_channels=position_embedding_dim+ESMC_embedding_dim,
                                              out_channels=n_filters,
                                              kernel_size=fs,
                                              padding='same')
                                    for fs in filter_sizes])
        if position_embedding_dim!=0:
            self.embedding = nn.Embedding(vocab_size, position_embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.Mish = nn.Mish()

    def forward(self, data1,data2):
        if  self.position_embedding_dim!=0 and self.ESMC_embedding_dim!=0:
            data1 = data1.to(dtype=torch.float32)
            embedded2 = self.embedding(data2)
            embedded = torch.cat((data1, embedded2), dim=-1)
            embedded = embedded.permute(0, 2, 1)
        elif self.position_embedding_dim!=0:
            embedded = self.embedding(data2.long())
            embedded = embedded.permute(0, 2, 1)
        else:
            embedded = data1.to(dtype=torch.float32)
            embedded = embedded.permute(0, 2, 1)
        conved = [self.Mish(conv(embedded)) for conv in self.convs]
        pooled = [F.max_pool1d(conv, math.ceil(conv.shape[2] // 10)) for conv in conved]
        flatten = [pool.view(pool.size(0), -1) for pool in pooled]
        cat = self.dropout(torch.cat(flatten, dim=1))
        return cat

class MLP(nn.Module):
    def __init__(self, vocab_size:int,position_embedding_dim:int,ESMC_embedding_dim:int,n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(len(filter_sizes) * n_filters * 10, len(filter_sizes) * n_filters * 5),
            nn.Mish(),
            nn.Dropout(),
            nn.Linear(len(filter_sizes) * n_filters * 5, len(filter_sizes) * n_filters),
            nn.Mish(),
            nn.Linear(len(filter_sizes) * n_filters, len(filter_sizes) * n_filters // 2),
            nn.Mish(),
        )
    def forward(self, cat):
        Representation = self.fc(cat)
        return Representation

class FC(nn.Module):
    def __init__(self, vocab_size: int, position_embedding_dim: int, ESMC_embedding_dim: int, n_filters: int,
                 filter_sizes: list, output_dim: int, dropout: float):
        super(FC, self).__init__()
        self.fc1 = nn.Linear(len(filter_sizes) * n_filters // 2, output_dim)

    def forward(self, Representation):
        logits = self.fc1(Representation)
        return logits

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ProjectionHead, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.fc2 = nn.Linear(out_dim, out_dim)

    def forward(self, Representation):
        x = F.relu(self.fc1(Representation))
        Feature = self.fc2(x)
        return Feature

from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
with torch.device(device):
    model_ESMC = ESMC(
        d_model=960,
        n_heads=15,
        n_layers=30,
        tokenizer=get_esmc_model_tokenizers(),
        use_flash_attn=True,
    ).eval()
    state_dict = torch.load("save/esmc_300m_2024_12_v0.pth", map_location=device, )
model_ESMC.load_state_dict(state_dict)
if device.type != "cpu":
    model_ESMC = model_ESMC.to(torch.bfloat16)