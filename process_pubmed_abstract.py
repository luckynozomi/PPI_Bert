import os
import csv

def my_sentence_split_func(Str):

    sentArr = []
    tmpSent = ""
    sentEnd = {'.', ';', '?'}
    current_start = 0
    for i in range(len(Str)):
        if Str[i] in sentEnd:
            if ((i + 2) >= len(Str)) or (Str[i + 1] == " " and Str[i + 2].isalnum()):
                sentArr.append(tmpSent)
                current_start = i + 2
                i = i + 1
                tmpSent = ""
        elif tmpSent == "" and Str[i] == " ":
            continue
        else:
            tmpSent = tmpSent + Str[i]
    if len(tmpSent) > 10:
        sentArr.append(tmpSent)
    return sentArr

INPUT_PATH = "pubmed18n0001PubMed_list.csv"
OUTPUT_PATH = ""

for file in os.listdir(INPUT_PATH):

    with open(os.path.join(OUTPUT_PATH, file), "w") as output_file:

        with open(os.path.join(INPUT_PATH, file), "r") as input_file:
            input_reader = csv.reader(input_file)
            input_reader.__next__()
            for row in input_reader:
                title = row[1]
                abstract = row[2]
                if title != "":
                    output_file.write(title + '\n\n')
                if abstract != "":
                    sent_iter = my_sentence_split_func(abstract)
                    for sent in sent_iter:
                        output_file.write(sent + '\n')
                    output_file.write('\n')
