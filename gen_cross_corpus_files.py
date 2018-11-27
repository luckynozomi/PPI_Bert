import os
import copy
from process_corpus import append, save_file

CORPORA = {
    "AImed": "AImed",
    "BioInfer": "BioInfer"
}

DEFAULT_DEV_CORPORA = {
    "BioInfer": "AImed",
    "AImed": "BioInfer"
}

def read_instances(corpora_path):

    with open(os.path.join(corpora_path, "dev.tsv")) as infile:
        ret = [line.strip() for line in infile]
    return ret


def copy_file(src_file, tgt_file):
    with open(src_file, "r") as src:
        with open(tgt_file, "w") as tgt:
            for line in src:
                tgt.write(line.strip() + '\n')


if __name__ == "__main__":

    for corpora_name, corpora_path in CORPORA.items():

        output_dir = corpora_name + '_CC'
        this_instances = read_instances(corpora_path)
        for BALA_METHOD in ["ORIG", "BALA"]:
            this_output_dir = os.path.join(output_dir, BALA_METHOD)
            if not os.path.exists(this_output_dir):
                os.makedirs(this_output_dir)
            result_instances = append(this_instances, BALA_METHOD)
            save_file(result_instances, os.path.join(this_output_dir, "train.tsv"))
            dev_name = DEFAULT_DEV_CORPORA[corpora_name]
            dev_path = os.path.join(CORPORA[dev_name], "dev.tsv")
            copy_file(dev_path, os.path.join(this_output_dir, "dev.tsv"))
