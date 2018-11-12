import os

CORPUS = "AImed"
SEP_METHOD = "SEP_SENT"
AUGMENT_METHOD = "ORIG"
OUTPUT_DIR = "ppi_lstm_data"

os.mkdir(OUTPUT_DIR)

for i in range(10):
    for file in ["train.tsv", "dev.tsv"]:
        path = os.path.join(CORPUS, SEP_METHOD, AUGMENT_METHOD, "fold_"+str(i), file)
        if file == "train.tsv":
            out_str = "BioInfer_tokenized_f" + str(i) + "_train.txt"
        else:
            out_str = "BioInfer_tokenized_f" + str(i) + "_test.txt"
        with open(path, "r") as input_file:
            with open(os.path.join(OUTPUT_DIR, out_str), "w") as output_file:
                for line in input_file:
                    output_file.write(line)
