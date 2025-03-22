import argparse
import numpy as np
import torch
from esm.models.esmc import ESMC
import torch.nn.functional as F
from matplotlib import pyplot as plt
from scipy.spatial import distance
from sklearn import manifold
from torch import sigmoid
from torch.utils.data import Dataset, DataLoader
from matplotlib.lines import Line2D
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
def loadtxt_test(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    sequences = []
    labels = []
    for i in range(0, len(lines), 2):
        label = lines[i][1:].strip()
        sequence = lines[i + 1].strip().upper()
        sequence = ''.join(['X' if aa not in amino_acids else aa for aa in sequence])
        labels.append(list(map(int, label)))
        sequences.append(sequence)
    return sequences, labels


def process_batch(model1, batch,max_length):
    model1.eval()
    with torch.no_grad():
        batch_input_ids = model1._tokenize(batch)
        current_length = batch_input_ids.shape[-1]
        padding_needed = max_length - current_length
        if padding_needed > 0:
            batch_input_ids = F.pad(batch_input_ids, (0, padding_needed), value=0)
        batch_esm_output = model1(batch_input_ids)
        batch_embeddings = batch_esm_output.embeddings
    return batch_embeddings


def seq_to_embed(sequences,amino_acids,max_length):
    amino_ids={amino_acids[i]:i for i in range(len(amino_acids))}
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [amino_ids[seq[j]] for j in range(len(seq))]
        if len(encoded_seq)<max_length:
            encoded_seq=[0]+encoded_seq+[0]*(max_length-len(encoded_seq)-1)
        else:
            encoded_seq=encoded_seq[:max_length]
        encoded_sequences.append(encoded_seq)
    encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long)
    return encoded_sequences

class ProteinDataset(Dataset):
    def __init__(self, train_embeddings,train_seq_embeddings, labels):
        self.train_embeddings = train_embeddings
        self.train_seq_embeddings = train_seq_embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.train_embeddings[idx],self.train_seq_embeddings[idx], self.labels[idx]

def get_config():
    # Parameter Management
    parse = argparse.ArgumentParser(description='Parameter Management')
    # Classification Task
    parse.add_argument('-peptides_name', type=list,
                       default=['AAP', 'ABP', 'ACP.txt', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP','AHP', 'AIP', 'AMRSAP',
                                'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP','THP'], help='Types of therapeutic peptides in the dataset')
    # Model Parameters
    parse.add_argument('-max_length', type=int, default=52, help='Maximum allowed length for sequence word encoding')
    parse.add_argument('-model', type=str, default='MCMFPP', help='Model selection: MCMFPP')
    parse.add_argument('-vocab_size', type=int, default=21, help='Vocabulary size')
    parse.add_argument('-position_embedding_dim', type=int, default=128,
                       help='Embedding dimension for positional encoding word vectors (128). If set to 0, positional encoding is not performed')
    parse.add_argument('-ESMC_embedding_dim', type=int, default=960, help='0 or 960. 0 indicates that ESMC extraction is not used')
    parse.add_argument('-n_filters', type=int, default=128,
                       help='Number of filters for each convolutional kernel')
    parse.add_argument('-filter_sizes', type=list, default=[3, 4, 5], help='Sizes of convolutional kernels')
    parse.add_argument('-dropout', type=float, default=0.6, help='Dropout rate')
    parse.add_argument('-batch_size', type=int, default=256, help='Batch size')

    # Dataset Loading Path
    parse.add_argument('-test_direction', type=str, default='../MFTP/test.txt', help='Location where the test set is stored')
    # Model Saving Path
    config = parse.parse_args()
    return config

if __name__ == "__main__":
    device='cuda'
    args = get_config()
    test_sample, test_label = loadtxt_test(args.test_direction)
    test_labels = torch.tensor(test_label, dtype=torch.float32)
    test_seq_embeddings = seq_to_embed(test_sample, amino_acids, args.max_length)
    model0 = ESMC.from_pretrained("esmc_300m")
    test_embeddings = []
    for i in range(0, len(test_sample), args.batch_size):
        batch = test_sample[i:i + args.batch_size]
        batch_embeddings = process_batch(model0, batch, args.max_length)
        test_embeddings.extend(batch_embeddings)
    test_dataset = ProteinDataset(test_embeddings, test_seq_embeddings, test_labels)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)

    model1 = torch.load('../save/classifier_cfec.pth')
    model2 = torch.load('../save/classifier_slfe.pth')
    print(model1)
    print(model2)

    head_label = [1, 2, 4, 6, 8, 9, 11, 12, 13, 15, 16, 17, 20]
    tail_label = [0, 3, 5, 7, 10, 14, 19, 18]
    label = []
    hidden1 = []
    hidden2 = []
    preds1 = []
    preds2 = []
    preds=[]
    with torch.no_grad():
        for batch in test_loader:
            data1, data2, labels = batch
            data1, data2, labels = data1.to(device), data2.to(device), labels.long().to(device)
            logits1,hidden1_fea = model1(data1, data2)
            logits2,hidden2_fea = model2(data1, data2)
            preds1_value, preds2_value = sigmoid(logits1), sigmoid(logits2)
            batch_size = preds1_value.shape[0]
            num_classes = preds1_value.shape[1]
            preds_value = torch.zeros((batch_size, num_classes), device=device)
            for i in head_label:
                preds_value[:, i] = 0.55 * preds1_value[:, i] + 0.45 * preds2_value[:, i]
            for k in tail_label:
                preds_value[:, k] = preds2_value[:, k]
            hidden1.extend(hidden1_fea.data.cpu().detach().numpy())
            hidden2.extend(hidden2_fea.data.cpu().detach().numpy())
            preds1.extend(preds1_value.data.cpu().detach().numpy())
            preds2.extend(preds2_value.data.cpu().detach().numpy())
            preds.extend(preds_value.data.cpu().detach().numpy())
            label.extend(labels.data.cpu().detach().numpy())

    categories=args.peptides_name
    func_labels = [[categories[j] for j in range(len(row)) if row[j] == 1] for row in label]
    joined_func_labels = ['_'.join(label) for label in func_labels]
    Max_Nums_Lab_Combin=25
    if Max_Nums_Lab_Combin != 25:
        unique_func_labels = set(joined_func_labels)
    else:
        unique_func_labels = ['THP', 'ADP', 'DPPIP', 'BBP', 'AHP', 'AIP',
                              'ADP_AIP', 'ACP', 'BIP', 'CPP', 'ADP_DPPIP','SBP',
                              'AAP', 'ABP_ACP_AFP', 'ATP', 'ABP', 'ABP_AFP_APP', 'AHP_DPPIP',
                              'APP', 'AFP','QSP', 'ABP_AFP', 'ABP_ACP', 'ACVP', 'AVP']

    case_labels =  ['THP', 'ADP', 'DPPIP', 'BBP', 'AHP', 'AIP',
                    'ADP_AIP', 'ACP', 'BIP', 'CPP', 'ADP_DPPIP','SBP',
                    'AAP', 'ABP_ACP_AFP', 'ATP', 'ABP', 'ABP_AFP_APP', 'AHP_DPPIP',
                    'APP', 'AFP', 'QSP', 'ABP_AFP', 'ABP_ACP', 'ACVP', 'AVP']


    case_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
    'black', '#7f7f7f', '#bcbd22', '#17becf', 'black', '#ffbb78',
    '#98df8a', 'black', '#c5b0d5', '#c49c94', 'black', 'black',
    '#dbdb8d', '#9edae5', '#393b79', 'black', 'black', '#7b4173','#637939'
]
    tag_labels1= ['THP', 'ADP', 'DPPIP', 'BBP', 'AHP', 'AIP',
                  'ACP', 'BIP', 'CPP','SBP',
                  'AAP', 'ATP', 'ABP',
                  'APP', 'AFP','QSP', 'ACVP', 'AVP','Others']
    tag_colors1 = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
        '#7f7f7f', '#bcbd22', '#17becf', '#ffbb78',
        '#98df8a', '#c5b0d5', '#c49c94',
        '#dbdb8d', '#9edae5', '#393b79', '#843c39', '#637939','black'
    ]

    legend_handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=tag_colors1[i],
                             markersize=15, label=label) for i, label in
                      enumerate(tag_labels1)]

    font = {'family': 'Times New Roman',
            'weight': 'normal',
            'size': 30, }
    out_name = ['hidden1','hidden2', 'preds1', 'preds2','preds']
    Intlabels = [int(''.join(map(str, lab.tolist())), 2) for lab in label]
    labels_unique_nums = len(np.unique(Intlabels))

    if Max_Nums_Lab_Combin <= 25:
        cmap = case_colors
    if tag_labels1:
        feat = np.array(preds)

        multifunc_nums = case_labels[-2]
        savepath = f'../Figures/mcmfpp_single_TSNE.png'

        plt.figure(figsize=(15, 12), dpi=600)
        feat_tsne = manifold.TSNE(n_components=2, learning_rate='auto', init="pca", perplexity=20,
                                  random_state=20230226).fit_transform(feat)
        plt.xlim([min(feat_tsne[:, 0] - 1), max(feat_tsne[:, 0] + 1)])
        plt.ylim([min(feat_tsne[:, 1] - 1), max(feat_tsne[:, 1] + 1)])

        # Locate the density center of each category
        density_centers = {}
        for label in np.unique(Intlabels):
            indices = np.where(Intlabels == label)[0]
            points = feat_tsne[indices]
            median_point = np.median(points, axis=0)
            dist_to_median = distance.cdist([median_point], points, 'euclidean')[0]
            medoid_index = indices[np.argmin(dist_to_median)]
            density_centers[label] = feat_tsne[medoid_index]

        centers = list(density_centers.values())
        labels = case_labels
        # Draw a scatter plot
        for j, (label, func_label) in enumerate(
                zip(np.unique(Intlabels), case_labels)):
            print('j,label,func_label:',j,label,func_label)
            plt.scatter(feat_tsne[Intlabels == label, 0], feat_tsne[Intlabels == label, 1], color=cmap[j],
                        label=func_label)
            if func_label not in tag_labels1:
                continue
            center = density_centers[label]
            if func_label=='DPPIP':
                text_x = center[0] + 5
                text_y = center[1] + 5
            elif func_label=='ACVP':
                text_x = center[0]
                text_y = center[1] - 8
            elif func_label=='ABP':
                text_x = center[0]
                text_y = center[1] - 8
            elif func_label=='ACP':
                text_x = center[0] + 8
                text_y = center[1] - 8
            elif func_label=='THP':
                text_x = center[0]
                text_y = center[1] + 8
            elif func_label=='AHP':
                text_x = center[0] -5
                text_y = center[1] -8
            elif func_label=='QSP':
                text_x = center[0]
                text_y = center[1] - 8
            elif func_label=='AVP':
                text_x = center[0] +5
                text_y = center[1] - 8
            elif func_label=='BBP':
                text_x = center[0] +10
                text_y = center[1] - 8
            elif func_label=='ATP':
                text_x = center[0]
                text_y = center[1] - 10
            else:
                text_x = center[0] - 5
                text_y = center[1] - 5

            plt.plot([center[0], text_x], [center[1], text_y], color='black', linewidth=1, linestyle='--')
            plt.text(text_x, text_y, func_label, fontsize=20, ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        plt.subplots_adjust(right=0.8)
        plt.xlabel('TSNE 1', font)
        plt.ylabel("TSNE 2", font)
        plt.legend(handles=legend_handles, loc='upper left', bbox_to_anchor=(1, 1),
                   fontsize=20)
        plt.savefig(savepath, dpi=600)