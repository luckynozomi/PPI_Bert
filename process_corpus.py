from lxml import etree as ET
import copy
import random
import os
import csv


DATA_PATH = {
    #"PICAD": "PPICorpora/picad_processed.xml",
    "AImed": "processed_corpus/AImed.tsv"
}


def get_sent_list(split_file_path):
    ret = []
    with open(split_file_path, "r") as split_file:
        for line in split_file:
            ret.append(line.split()[1])
    return ret


def get_sents(split_file_path, data_path):
    ret = []
    sent_list = get_sent_list(split_file_path)

    with open(data_path, "r") as data_file:
        for line in data_file:
            line = line.strip()
            items = line.split()
            if any([items[0].split('.')[1] == sent_list_item.split('.')[1] for sent_list_item in sent_list]):
                ret.append(line)

    return ret


def save_file(data_list, file_name):
    with open(file_name, "w") as f:
        for instance in data_list:
            f.write(instance + '\n')


for data_name, data_path in DATA_PATH.items():
    tenfold_splits_dir = os.path.join("Tenfold_splits", data_name)
    for split_method in ["BALA", "ORIG"]:
        for fold_num in range(10):
            fold_dir = "fold_"+str(fold_num)
            out_dir = os.path.join(data_name, split_method, fold_dir)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            train_split_file = os.path.join(tenfold_splits_dir, "train-"+str(fold_num+1))
            test_split_file = os.path.join(tenfold_splits_dir, "test-"+str(fold_num+1))
            train_sents = get_sents(train_split_file, data_path)
            test_sents = get_sents(test_split_file, data_path)

            save_file(train_sents, os.path.join(out_dir, "train.tsv"))
            save_file(test_sents, os.path.join(out_dir, "dev.tsv"))
