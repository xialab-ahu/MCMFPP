from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from esm.models.esmc import ESMC
from esm.tokenization import get_esmc_model_tokenizers
from models import *
from utils import *
amino_acids = "XACDEFGHIKLMNPQRSTVWY"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    if args.ESMC_embedding_dim != 0 and args.position_embedding_dim != 0:
        args.embed_type = "======Enabled===ESMC Encoding===and===Positional Encoding======"
    elif args.ESMC_embedding_dim != 0 and args.position_embedding_dim == 0:
        args.embed_type = "======Only===ESMC Encoding===is Enabled======"
    elif args.ESMC_embedding_dim == 0 and args.position_embedding_dim != 0:
        args.embed_type = "======Only===Positional Encoding===is Enabled======"
    else:
        print("No encoding format is enabled!!!!!!!!")
        print("Please check the values of position_embedding_dim and ESMC_embedding_dim!!!!!!!!")
        return None
    train_path = args.train_direction
    test_path = args.test_direction
    train_sample, train_label = loadtxt(train_path)
    test_sample, test_label = loadtxt(test_path)
    train_sample_slfe, train_label_slfe=train_sample, train_label
    train_labels_slfe = torch.tensor(train_label_slfe, dtype=torch.float32)
    test_labels = torch.tensor(test_label, dtype=torch.float32)
    if args.position_embedding_dim != 0:
        train_seq_embeddings_slfe = seq_to_embed(train_sample_slfe, amino_acids, args.max_length)
        test_seq_embeddings = seq_to_embed(test_sample, amino_acids, args.max_length)
    else:
        train_seq_embeddings_slfe = [0] * len(train_sample_slfe)
        test_seq_embeddings = [0] * len(test_sample)
    if args.ESMC_embedding_dim != 0:
        print("Loading the large model ESMC...")
        # model1 = ESMC.from_pretrained("esmc_300m")
        model1=model_ESMC
        print("The large model ESMC has been loaded successfully.")
        # Process training data
        train_embeddings_slfe = []
        for i in range(0, len(train_sample_slfe), args.batch_size):
            batch = train_sample_slfe[i:i + args.batch_size]
            batch_embeddings = process_batch(model1, batch, args.max_length)
            train_embeddings_slfe.extend(batch_embeddings.to('cpu'))
        print("SLFE data embedding with the large language model is complete.")
        # Process testing data
        test_embeddings = []
        for i in range(0, len(test_sample), args.batch_size):
            batch = test_sample[i:i + args.batch_size]
            batch_embeddings = process_batch(model1, batch, args.max_length)
            test_embeddings.extend(batch_embeddings.to('cpu'))
    else:
        train_embeddings_slfe = [0] * len(train_sample_slfe)
        test_embeddings = [0] * len(test_sample)
    # Create datasets
    train_dataset_slfe = ProteinDataset(train_embeddings_slfe, train_seq_embeddings_slfe, train_labels_slfe)
    test_dataset = ProteinDataset(test_embeddings, test_seq_embeddings, test_labels)

    # Create DataLoaders
    train_loader_slfe = DataLoader(dataset=train_dataset_slfe, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader_slfe, test_loader

def get_cfec_input(args):
    if args.ESMC_embedding_dim != 0 and args.position_embedding_dim != 0:
        args.embed_type = "======Enabled===ESMC Encoding===and===Positional Encoding======"
    elif args.ESMC_embedding_dim != 0 and args.position_embedding_dim == 0:
        args.embed_type = "======Only===ESMC Encoding===is Enabled======"
    elif args.ESMC_embedding_dim == 0 and args.position_embedding_dim != 0:
        args.embed_type = "======Only===Positional Encoding===is Enabled======"
    else:
        print("No encoding format is enabled!!!!!!!!")
        print("Please check the values of position_embedding_dim and ESMC_embedding_dim!!!!!!!!")
        return None
    train_path = args.train_direction
    test_path = args.test_direction
    train_sample, train_label = loadtxt(train_path)
    test_sample, test_label = loadtxt(test_path)
    train_sample_cfec, train_label_cfec = CFEC.augment_peptide_dataset_with_reversals(train_sample, train_label, args.N)
    train_labels_cfec = torch.tensor(train_label_cfec, dtype=torch.float32)
    test_labels = torch.tensor(test_label, dtype=torch.float32)
    if args.position_embedding_dim != 0:
        train_seq_embeddings_cfec = seq_to_embed(train_sample_cfec, amino_acids, args.max_length)
        test_seq_embeddings = seq_to_embed(test_sample, amino_acids, args.max_length)
    else:
        train_seq_embeddings_cfec = [0] * len(train_sample_cfec)
        test_seq_embeddings = [0] * len(test_sample)
    if args.ESMC_embedding_dim != 0:
        print("Loading the large model ESMC...")
        # model1 = ESMC.from_pretrained("esmc_300m")
        model1=model_ESMC
        print("The large model ESMC has been loaded successfully.")
        # Process training data
        train_embeddings_cfec = []
        for i in range(0, len(train_sample_cfec), args.batch_size):
            batch = train_sample_cfec[i:i + args.batch_size]
            batch_embeddings = process_batch(model1, batch, args.max_length)
            train_embeddings_cfec.extend(batch_embeddings.to('cpu'))
        print("CFEC data embedding with the large language model is complete.")
        # Process testing data
        test_embeddings = []
        for i in range(0, len(test_sample), args.batch_size):
            batch = test_sample[i:i + args.batch_size]
            batch_embeddings = process_batch(model1, batch, args.max_length)
            test_embeddings.extend(batch_embeddings.to('cpu'))
    else:
        train_embeddings_cfec = [0] * len(train_sample_cfec)
        test_embeddings = [0] * len(test_sample)
    # Create DataLoaders
    train_dataset_cfec = ProteinDataset(train_embeddings_cfec, train_seq_embeddings_cfec, train_labels_cfec)
    test_dataset = ProteinDataset(test_embeddings, test_seq_embeddings, test_labels)
    train_loader_cfec = DataLoader(dataset=train_dataset_cfec , batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)
    return  train_loader_cfec, test_loader


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
