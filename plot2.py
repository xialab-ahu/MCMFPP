import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np

# 类别标签
labels = ['AAP', 'ABP', 'ACP', 'ACVP', 'ADP', 'AEP', 'AFP', 'AHIVP', 'AHP', 'AIP', 'AMRSAP', 'APP', 'ATP', 'AVP', 'BBP', 'BIP', 'CPP', 'DPPIP', 'QSP', 'SBP', 'THP']

# PSCFA
pscfa_peptides_info = [
["AAP", 4, 1947, 14, 4],
["ABP", 341, 1472, 78, 78],
["ACP", 101, 1731, 92, 45],
["ACVP", 12, 1936, 16, 5],
["ADP", 29, 1839, 80, 21],
["AEP", 0, 1955, 10, 4],
["AFP", 145, 1640, 128, 56],
["AHIVP", 4, 1939, 15, 11],
["AHP", 154, 1719, 31, 65],
["AIP", 412, 1400, 13, 144],
["AMRSAP", 0, 1941, 21, 7],
["APP", 11, 1899, 50, 9],
["ATP", 26, 1896, 34, 13],
["AVP", 87, 1806, 56, 20],
["BBP", 1, 1941, 24, 3],
["BIP", 40, 1899, 20, 10],
["CPP", 57, 1856, 36, 20],
["DPPIP", 23, 1886, 40, 20],
["QSP", 40, 1913, 9, 7],
["SBP", 7, 1946, 8, 8],
["THP", 90, 1804, 30, 45]
]


mcmfpp_peptides_info = [
["AAP", 4, 1949, 14, 2],
["ABP", 342, 1470, 77, 80],
["ACP", 101, 1734, 92, 42],
["ACVP", 11, 1940, 17, 1],
["ADP", 32, 1844, 77, 16],
["AEP", 0, 1959, 10, 0],
["AFP", 148, 1635, 125, 61],
["AHIVP", 0, 1950, 19, 0],
["AHP", 161, 1733, 24, 51],
["AIP", 414, 1401, 11, 143],
["AMRSAP", 5, 1940, 16, 8],
["APP", 22, 1904, 39, 4],
["ATP", 36, 1903, 24, 6],
["AVP", 91, 1811, 52, 15],
["BBP", 5, 1942, 20, 2],
["BIP", 39, 1903, 21, 6],
["CPP", 57, 1856, 36, 20],
["DPPIP", 29, 1889, 34, 17],
["QSP", 42, 1917, 7, 3],
["SBP", 8, 1953, 7, 1],
["THP", 91, 1815, 29, 34]
]


# ETFC
etfc_peptides_info =  [
["AAP", 4, 1944, 14, 7],
["ABP", 333, 1460, 86, 90],
["ACP", 115, 1693, 78, 83],
["ACVP", 11, 1934, 17, 7],
["ADP", 26, 1841, 83, 19],
["AEP", 0, 1959, 10, 0],
["AFP", 158, 1608, 115, 88],
["AHIVP", 5, 1941, 14, 9],
["AHP", 149, 1730, 36, 54],
["AIP", 409, 1388, 16, 156],
["AMRSAP", 4, 1935, 17, 13],
["APP", 21, 1890, 40, 18],
["ATP", 27, 1893, 33, 16],
["AVP", 95, 1792, 48, 34],
["BBP", 4, 1941, 21, 3],
["BIP", 41, 1899, 19, 10],
["CPP", 49, 1855, 44, 21],
["DPPIP", 29, 1885, 34, 21],
["QSP", 39, 1915, 10, 5],
["SBP", 6, 1950, 9, 4],
["THP", 90, 1800, 30, 49]
]

sen_pscfa = []
spe_pscfa = []
for info in pscfa_peptides_info:
    tp, tn, fn, fp = info[1], info[2], info[3], info[4]
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    sen_pscfa.append(sen)
    spe_pscfa.append(spe)

sen_mcmfpp = []
spe_mcmfpp = []
for info in mcmfpp_peptides_info:
    tp, tn, fn, fp = info[1], info[2], info[3], info[4]
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    sen_mcmfpp.append(sen)
    spe_mcmfpp.append(spe)

sen_etfc = []
spe_etfc = []
for info in etfc_peptides_info:
    tp, tn, fn, fp = info[1], info[2], info[3], info[4]
    sen = tp / (tp + fn) if (tp + fn) > 0 else 0
    spe = tn / (tn + fp) if (tn + fp) > 0 else 0
    sen_etfc.append(sen)
    spe_etfc.append(spe)

x = np.arange(len(labels))
width = 0.25
colors = ['#999999', '#ffd966', '#1f77b4']
fig, ax = plt.subplots(figsize=(12, 3))
rects1 = ax.bar(x - width, sen_etfc, width, label='ETFC',color=colors[0])
rects2 = ax.bar(x , sen_pscfa, width, label='PSCFA',color=colors[1])
rects3 = ax.bar(x + width, sen_mcmfpp, width, label='MCMFPP',color=colors[2])
ax.set_ylabel('Sensitivity (SEN)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
plt.ylim(0, 1.12)
ax.legend(loc='center left', bbox_to_anchor=(0.3, 0.93), ncol=4)
fig.tight_layout()
plt.show()


fig, ax = plt.subplots(figsize=(12, 3))
rects1 = ax.bar(x - width, spe_etfc, width, label='ETFC',color=colors[0])
rects2 = ax.bar(x , spe_pscfa, width, label='PSCFA',color=colors[1])
rects3 = ax.bar(x + width, spe_mcmfpp, width, label='MCMFPP',color=colors[2])
ax.set_ylabel('Specificity (SPE)')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.legend(loc='center left', bbox_to_anchor=(0.3, 0.9), ncol=4)
ax.set_ylim(0.87, 1.03)
fig.tight_layout()
plt.show()

f1_pscfa = []
for i in range(len(sen_pscfa)):
    precision = pscfa_peptides_info[i][1] / (pscfa_peptides_info[i][1] + pscfa_peptides_info[i][4]) if (pscfa_peptides_info[i][1] + pscfa_peptides_info[i][4]) > 0 else 0
    f1 = 2 * precision * sen_pscfa[i] / (precision + sen_pscfa[i]) if (precision + sen_pscfa[i]) > 0 else 0
    f1_pscfa.append(f1)

f1_mcmfpp = []
for i in range(len(sen_mcmfpp)):
    precision = mcmfpp_peptides_info[i][1] / (mcmfpp_peptides_info[i][1] + mcmfpp_peptides_info[i][4]) if (mcmfpp_peptides_info[i][1] + mcmfpp_peptides_info[i][4]) > 0 else 0
    f1 = 2 * precision * sen_mcmfpp[i] / (precision + sen_mcmfpp[i]) if (precision + sen_mcmfpp[i]) > 0 else 0
    f1_mcmfpp.append(f1)

f1_etfc = []
for i in range(len(sen_etfc)):
    precision = etfc_peptides_info[i][1] / (etfc_peptides_info[i][1] + etfc_peptides_info[i][4]) if (etfc_peptides_info[i][1] + etfc_peptides_info[i][4]) > 0 else 0
    f1 = 2 * precision * sen_etfc[i] / (precision + sen_etfc[i]) if (precision + sen_etfc[i]) > 0 else 0
    f1_etfc.append(f1)

# Macro-F1
macro_f1_pscfa = np.mean(f1_pscfa)
macro_f1_mcmfpp = np.mean(f1_mcmfpp)
macro_f1_etfc = np.mean(f1_etfc)

# Micro-F1
tp_sum_pscfa = np.sum([info[1] for info in pscfa_peptides_info])
fp_sum_pscfa = np.sum([info[4] for info in pscfa_peptides_info])
fn_sum_pscfa = np.sum([info[3] for info in pscfa_peptides_info])
precision_micro_pscfa = tp_sum_pscfa / (tp_sum_pscfa + fp_sum_pscfa) if (tp_sum_pscfa + fp_sum_pscfa) > 0 else 0
recall_micro_pscfa = tp_sum_pscfa / (tp_sum_pscfa + fn_sum_pscfa) if (tp_sum_pscfa + fn_sum_pscfa) > 0 else 0
micro_f1_pscfa = 2 * precision_micro_pscfa * recall_micro_pscfa / (precision_micro_pscfa + recall_micro_pscfa) if (precision_micro_pscfa + recall_micro_pscfa) > 0 else 0

tp_sum_mcmfpp = np.sum([info[1] for info in mcmfpp_peptides_info])
fp_sum_mcmfpp = np.sum([info[4] for info in mcmfpp_peptides_info])
fn_sum_mcmfpp = np.sum([info[3] for info in mcmfpp_peptides_info])
precision_micro_mcmfpp = tp_sum_mcmfpp / (tp_sum_mcmfpp + fp_sum_mcmfpp) if (tp_sum_mcmfpp + fp_sum_mcmfpp) > 0 else 0
recall_micro_mcmfpp = tp_sum_mcmfpp / (tp_sum_mcmfpp + fn_sum_mcmfpp) if (tp_sum_mcmfpp + fn_sum_mcmfpp) > 0 else 0
micro_f1_mcmfpp = 2 * precision_micro_mcmfpp * recall_micro_mcmfpp / (precision_micro_mcmfpp + recall_micro_mcmfpp) if (precision_micro_mcmfpp + recall_micro_mcmfpp) > 0 else 0

tp_sum_etfc = np.sum([info[1] for info in etfc_peptides_info])
fp_sum_etfc = np.sum([info[4] for info in etfc_peptides_info])
fn_sum_etfc = np.sum([info[3] for info in etfc_peptides_info])
precision_micro_etfc = tp_sum_etfc / (tp_sum_etfc + fp_sum_etfc) if (tp_sum_etfc + fp_sum_etfc) > 0 else 0
recall_micro_etfc = tp_sum_etfc / (tp_sum_etfc + fn_sum_etfc) if (tp_sum_etfc + fn_sum_etfc) > 0 else 0
micro_f1_etfc = 2 * precision_micro_etfc * recall_micro_etfc / (precision_micro_etfc + recall_micro_etfc) if (precision_micro_etfc + recall_micro_etfc) > 0 else 0

print("Model comparison under macro F1 and micro F1 scores")
print("PSCFA Macro-F1:", macro_f1_pscfa)
print("PSCFA Micro-F1:", micro_f1_pscfa)
print("MCMFPP Macro-F1:", macro_f1_mcmfpp)
print("MCMFPP Micro-F1:", micro_f1_mcmfpp)
print("ETFC Macro-F1:", macro_f1_etfc)
print("ETFC Micro-F1:", micro_f1_etfc)


fig, ax = plt.subplots(figsize=(12, 3))
rects1 = ax.bar(x - width, f1_etfc, width, label='ETFC',color=colors[0])
rects2 = ax.bar(x , f1_pscfa, width, label='PSCFA',color=colors[1])
rects3 = ax.bar(x + width, f1_mcmfpp, width, label='MCMFPP',color=colors[2])
ax.set_ylabel('F1 Score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0)
ax.legend(loc='center left', bbox_to_anchor=(0.3, 0.9), ncol=4)
ax.set_ylim(0, 1.01)
fig.tight_layout()
plt.show()