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
    parse.add_argument('-random_seed',type=int, default=20230226 ,help='Random seed (20230226, same as ETFC)')
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
    parse.add_argument('-train_direction', type=str, default='MFTP/train.txt', help='Location of the training set')
    parse.add_argument('-test_direction', type=str, default='MFTP/test.txt', help='Location of the test set')
    # Save the results of MCMFPP peptide prediction
    parse.add_argument('-mcmfpp_predictions', type=str, default='save/mcmfpp_predictions.txt', help='Save the results of MCMFPP peptide prediction')
    config = parse.parse_args()
    return config

args = get_config()
head_label = [1, 2, 4, 6, 8, 9, 11, 12, 13, 15, 16, 17, 20]
tail_label = [0, 3, 5, 7, 10, 14, 19,18]
mcmfpp=MCMFPP(head_label, tail_label,weight1=args.W1, weight2=args.W2)
classifier_SLFE=torch.load('save/classifier_slfe.pth')
classifier_CFEC=torch.load('save/classifier_cfec.pth')
_,test_loader=get_slfe_input(args)
preds,reals=mcmfpp.predict(classifier_SLFE,classifier_CFEC,test_loader)
np.savetxt('save/mcmfpp_preds.csv', preds, delimiter=',')
np.savetxt('save/reals.csv', reals, delimiter=',')

label_combination_indices = {}
for i, label in enumerate(reals):
    s = []
    for l in range(len(label)):
        if label[l] == 1:
            s.append(args.peptides_name[l])
    s_tuple = tuple(s)
    if s_tuple in label_combination_indices:
        label_combination_indices[s_tuple].append(i)
    else:
        label_combination_indices[s_tuple] = [i]

key_length_list = [(key, len(value)) for key, value in label_combination_indices.items()]
sorted_key_length_list = sorted(key_length_list, key=lambda x: x[1], reverse=True)
final_label_combination_indices = {}
for key, _ in sorted_key_length_list[:35]:
    final_label_combination_indices[key] = label_combination_indices[key]
# print(final_label_combination_indices)
# key_length_list = [(key, len(value)) for key, value in final_label_combination_indices.items()]
# for key, length in key_length_list:
#     print(f"Key name (label combination): {key}, length of the key (length of the sample index list): {length}")
print("===================================================================================")
for key, value in final_label_combination_indices.items():
    label_combination_score = evaluate(preds[value], reals[value], threshold1=args.threshold1,threshold2=args.threshold2)
    print(f"=====Label Combination: {key}=====Performance Evaluation======")
    print(f'coverage: {label_combination_score[1]:.3f}')
    print(f'accuracy: {label_combination_score[2]:.3f}')
    print(f'absolute_true: {label_combination_score[3]:.3f}')

predicted_labels = obtain_functional_predictions(preds, args.threshold1, args.threshold2)
for i, peptide_name in enumerate(args.peptides_name):
    pred_class = predicted_labels[:, i]
    real_class = reals[:, i]
    tp = np.sum((pred_class == 1) & (real_class == 1))
    tn = np.sum((pred_class == 0) & (real_class == 0))
    fp = np.sum((pred_class == 1) & (real_class == 0))
    fn = np.sum((pred_class == 0) & (real_class == 1))
    print(f"====={peptide_name}=====  TP: {tp}, TN: {tn}, FP: {fp}, FN: {fn}")
    print()

print("===================================================================================")
score = evaluate(preds, reals, threshold1=args.threshold1, threshold2=args.threshold2)
print("=====MCMFPP=====Test=====Performance Evaluation======")
print(f'aiming: {score[0]:.3f}')
print(f'coverage: {score[1]:.3f}')
print(f'accuracy: {score[2]:.3f}')
print(f'absolute_true: {score[3]:.3f}')
print(f'absolute_false: {score[4]:.3f}\n')


# The functional prediction results of MCMFPP
test_sample, _ = loadtxt(args.test_direction)
functional_predictions=obtain_functional_predictions(preds,args.threshold1,args.threshold2)
functional_predictions_result = [[args.peptides_name[i] for i, value in enumerate(row) if value == 1] for row in functional_predictions]
with open("save\mcmfpp_functional prediction_output.txt", "w", encoding="utf-8") as file:
    for seq, prediction in zip(test_sample, functional_predictions_result):
        file.write('>' + seq + "\n")
        prediction_str = ", ".join(prediction)
        file.write(prediction_str  + "\n")
print("The prediction results for the functional sequences in the test set have been saved to: save1\mcmfpp_functional prediction_output.txt." )