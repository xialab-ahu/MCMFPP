# The statistical test of MCMFPP

## Related Files

| FILE NAME                 | DESCRIPTION                                      |
|:--------------------------|:-------------------------------------------------|
| five_subsets              | Independent test set split into five subsets     |
| ten-fold cross-validation | Script and data files for 10-fold data splitting |
| preprocess.py             | Data preprocessing, encoding, and loading        |
| models.py                 | models related to MCMFPP                         |
| train_slfe1.py            | Training of the Sub-classifier SLFE              |
| train_cfec1.py            | Training of the Sub-classifier CFEC              |
| predict_MCMFPP1.py        | Prediction with the Fusion Classifier MCMFPP     |
| estimate.py               | evaluation metrics for prediction                |
| utils.py                  | Some functions that will be used during training |
| save1                     | Save model weights and predictions               |

## Running the Statistical Significance Tests  
If you have successfully set up the environment as instructed in the README, please run the following command to reproduce our statistical significance testing experiments.
```bash
activate mcmfpp
```
Change the directory to MCMFPP-main\Statistical_test using the cd command, or use the absolute path of MCMFPP-main.
```
cd Statistical_test
```
We have provided the trained model weights of the sub-classifiers SLFE and CFEC, which are saved in the save1 directory.You can run the following command to obtain the experimental results on the training and test sets.
```bash
python predict_MCMFPP1.py
```
You can run the following commands to perform statistical significance tests on the experimental results:
```bash
python statistical_test_on_ten-fold-cross-validation.py
```
```bash
python statistical_test_in_subtests.py
```
If you wish to retrain the sub-classifiers, you can use the following commands to perform 10-fold cross-validation and obtain the corresponding model weights (this process may take approximately 10–15 hours).
```bash
python train_slfe1.py
```
```bash
python train_cfec1.py
```
These scripts will perform 10-fold training on the SLFE and CFEC models and save the resulting weights to the save1 directory.