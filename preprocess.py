import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from esm.models.esmc import ESMC
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
def loadtxt(file_path):
    with open(file_path, "r") as file:
        lines = file.readlines()
    sequences = []
    labels = []
    for i in range(0, len(lines), 2):
        label = lines[i][1:].strip()
        sequence = lines[i + 1].strip().upper()
        if any(aa not in amino_acids for aa in sequence):
            continue  # 跳过包含非标准氨基酸的序列
        labels.append(list(map(int, label)))
        sequences.append(sequence)
    return sequences, labels

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
    # 将列表转换为Tensor
    encoded_sequences = torch.tensor(encoded_sequences, dtype=torch.long)
    return encoded_sequences

def process_batch(model1, batch,max_length):
    model1.eval()
    with torch.no_grad():
        batch_input_ids = model1._tokenize(batch)
        # 计算每个序列需要填充的长度
        current_length = batch_input_ids.shape[-1]
        padding_needed = max_length - current_length
        # 如果需要填充，计算填充的维度
        if padding_needed > 0:
            # 填充维度 (0, 0, ..., 0) 表示在最后一个维度（序列长度）的末尾填充
            batch_input_ids = F.pad(batch_input_ids, (0, padding_needed), value=0)
        batch_esm_output = model1(batch_input_ids)
        batch_embeddings = batch_esm_output.embeddings
    return batch_embeddings

class ProteinDataset(Dataset):
    def __init__(self, train_embeddings,train_seq_embeddings, labels):
        self.train_embeddings = train_embeddings
        self.train_seq_embeddings = train_seq_embeddings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        return self.train_embeddings[idx],self.train_seq_embeddings[idx], self.labels[idx]

def get_slfe_input(args):
    if args.ESMC_embedding_dim !=0 and args.position_embedding_dim !=0:
        args.embed_type="======启用===ESMC编码===和===数字编码======"
    elif args.ESMC_embedding_dim !=0 and args.position_embedding_dim ==0:
        args.embed_type = "======只启用===ESMC编码======"
    elif args.ESMC_embedding_dim == 0 and args.position_embedding_dim != 0:
        args.embed_type = "======只启用===数字编码======"
    else:
        print("未启用任何编码格式!!!!!!!!")
        print("请检查position_embedding_dim和ESMC_embedding_dim大小!!!!!!!!")
        return None
    test_path = args.test_direction
    test_sample, test_label = loadtxt(test_path)
    test_labels = torch.tensor(test_label, dtype=torch.float32)
    if args.position_embedding_dim != 0:
        test_seq_embeddings = seq_to_embed(test_sample, amino_acids, args.max_length)
    else:
        test_seq_embeddings = [0] * len(test_sample)
    if args.ESMC_embedding_dim != 0:
        # 准备数据
        print("加载大模型ESMC中")
        model1 = ESMC.from_pretrained("esmc_300m")
        print("大模型ESMC已加载完成")
        # 处理测试数据
        test_embeddings = []
        for i in range(0, len(test_sample), args.batch_size):
            batch = test_sample[i:i + args.batch_size]
            batch_embeddings = process_batch(model1, batch, args.max_length)
            test_embeddings.extend(batch_embeddings.to('cpu'))
    else:
        test_embeddings = [0] * len(test_sample)
    # 创建数据集
    test_dataset = ProteinDataset(test_embeddings, test_seq_embeddings, test_labels)
    # 创建DataLoader
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return  test_loader



"""ESMC is from
@software{evolutionaryscale_2024,
  author = {{EvolutionaryScale Team}},
  title = {evolutionaryscale/esm},
  year = {2024},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.14219303},
  URL = {https://doi.org/10.5281/zenodo.14219303}
}
"""