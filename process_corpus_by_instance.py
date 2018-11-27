from sklearn.model_selection import StratifiedKFold
import numpy as np
import os


DATA_PATH = {
    #"PICAD": "PPICorpora/picad_processed.xml",
    "AImed": "processed_corpus/AImed.tsv"
}


def save_file(data_list, file_name):
    with open(file_name, "w") as f:
        for instance in data_list:
            f.write(instance + '\n')


def read_corpus(data_path):
    sents = []
    labels = []
    with open(data_path, "r") as data_file:
        for line in data_file:
            line = line.strip()
            sents.append(line)
            labels.append(int(line.split()[1].strip()=="P"))
    return sents, np.asarray(labels)


for data_name, data_path in DATA_PATH.items():

    sents, labels = read_corpus(data_path)
    strkfold = StratifiedKFold(n_splits=10)
    folds = strkfold.split(np.zeros(len(labels)), labels)
    for split_method in ["BALA", "ORIG"]:
        fold_num = 0
        for train_idx, test_idx in list(folds):
            fold_dir = "fold_"+str(fold_num)
            out_dir = os.path.join(data_name+"_split_ins", split_method, fold_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            train_sents = [sents[i] for i in train_idx]
            test_sents = [sents[i] for i in test_idx]
            save_file(train_sents, os.path.join(out_dir, "train.tsv"))
            save_file(test_sents, os.path.join(out_dir, "dev.tsv"))
            fold_num += 1
