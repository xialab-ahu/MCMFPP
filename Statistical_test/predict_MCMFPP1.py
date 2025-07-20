import argparse
from models import *
from estimate import *
from preprocess import *
def get_config():
    parse = argparse.ArgumentParser(description='Parameter Management')
    #Classification Task
    parse.add_argument('-task', type=str, default='MFTP', help='Task type selection: MFTP')
    parse.add_argument('-peptides_name', type=list, default=['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP',
       'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP','BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP'], help='Types of therapeutic peptides in the dataset')
    parse.add_argument('-random_seed',type=int, default=20230226 ,help='Random seed (20230226, same as PSCFA)')
    parse.add_argument('-criterion', type=str, default='FDL', help='Loss function selection: FDL')
    parse.add_argument('-max_length', type=int, default=52, help='Maximum allowed length for sequence encoding')
    parse.add_argument('-model', type=str, default='mcmfpp', help='Model selection: MCMFPP')
    parse.add_argument('-vocab_size', type=int, default=21, help='Vocabulary size')
    parse.add_argument('-position_embedding_dim', type=int, default=128, help='128; if 0, no positional encoding is applied')
    parse.add_argument('-ESMC_embedding_dim', type=int, default=960, help='0 or 960 (0 indicates no ESMC extraction)')
    parse.add_argument('-n_filters', type=int, default=128, help='Number of filters per convolution kernel')
    parse.add_argument('-filter_sizes', type=list, default=[3, 4, 5], help='Sizes of convolution kernels')
    parse.add_argument('-dropout', type=float, default=0.6, help='Dropout rate')
    parse.add_argument('-epochs', type=int, default=200, help='epochs')
    parse.add_argument('-N', type=int, default=2, help='Augmentation multiplier (0-5)')
    parse.add_argument('-lr_slfe', type=float, default=0.0012, help='Learning rate for SLFE')
    parse.add_argument('-lr_cfec', type=float, default=0.0007, help='Learning rate for CFEC')
    parse.add_argument('-threshold1', type=float, default=0.6, help='Threshold-1')
    parse.add_argument('-threshold2', type=float, default=0.4, help='Threshold-2')
    parse.add_argument('-batch_size', type=int, default=256, help='Maximum Batch Size')
    #MCMFPP Model Fusion Parameters
    parse.add_argument('-W1', type=float, default=0.55, help='Weight for the CFEC sub-classification prediction head')
    parse.add_argument('-W2', type=float, default=0.4, help='Weight for the CFEC sub-classification prediction tail')
    #Dataset Loading Paths
    parse.add_argument('-train_direction', type=str, default='ten-fold cross-validation/sub_train_10-fold_9.txt',
                       help='Location of the training set')
    parse.add_argument('-test_direction', type=str, default='ten-fold cross-validation/sub_val_10-fold_9.txt',
                       help='Location of the test set')
    config = parse.parse_args()
    return config

args = get_config()
head_label = [1, 2, 4, 6, 8, 9, 11, 12, 13, 15, 16, 17, 20]
tail_label = [0, 3, 5, 7, 10, 14, 19,18]
mcmfpp=MCMFPP(head_label, tail_label,weight1=args.W1, weight2=args.W2)
for fold in [1,2,3,4,5,6,7,8,9,10]:
    classifier_SLFE = torch.load(f'save1/classifier_slfe_10-fold_{fold}.pth')
    classifier_CFEC = torch.load(f'save1/classifier_cfec_10-fold_{fold}.pth')
    args.train_direction=f'ten-fold cross-validation/sub_train_10-fold_{fold}.txt'
    args.test_direction=f'ten-fold cross-validation/sub_val_10-fold_{fold}.txt'
    _, test_loader = get_slfe_input(args)
    preds, reals = mcmfpp.predict(classifier_SLFE, classifier_CFEC, test_loader)
    print("===================================================================================")
    score = evaluate(preds, reals, threshold1=args.threshold1, threshold2=args.threshold2)
    print(f"=====MCMFPP=====val_fold_{fold}=====Performance Evaluation======")
    print(f'aiming: {score[0]:.3f}')
    print(f'coverage: {score[1]:.3f}')
    print(f'accuracy: {score[2]:.3f}')
    print(f'absolute_true: {score[3]:.3f}')
    print(f'absolute_false: {score[4]:.3f}\n')

for j in [1,2,3,4,5]:#j denotes the test subset
    classifier_SLFE = torch.load(f'../save/classifier_slfe.pth')
    classifier_CFEC = torch.load(f'../save/classifier_cfec.pth')
    args.train_direction=f'../MFTP/train.txt'
    args.test_direction=f'five_subsets/sub_test_{j}.txt'
    _, test_loader = get_slfe_input(args)
    preds, reals = mcmfpp.predict(classifier_SLFE, classifier_CFEC, test_loader)
    print("===================================================================================")
    score = evaluate(preds, reals, threshold1=args.threshold1, threshold2=args.threshold2)
    print(f"=====MCMFPP=====sub_test_{j}=====Performance Evaluation======")
    print(f'aiming: {score[0]:.3f}')
    print(f'coverage: {score[1]:.3f}')
    print(f'accuracy: {score[2]:.3f}')
    print(f'absolute_true: {score[3]:.3f}')
    print(f'absolute_false: {score[4]:.3f}\n')