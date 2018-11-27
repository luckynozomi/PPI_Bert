import csv
import copy

result_file = "/Users/xinsui/Dropbox/PPI_Bert/0test_results.tsv"


def calc_f1(pred_labels, true_labels):
    t_p = sum([pred_labels[i] * true_labels[i] for i in range(len(pred_labels))])
    true = sum(true_labels)
    positive = sum(pred_labels)
    try:
        precision = t_p / positive
        recall = t_p / true
        f_1 = precision * recall * 2 / (precision + recall)
        return f_1
    except ZeroDivisionError:
        return 0


probs = []
true_labels = []
with open(result_file, "r") as result_FILE:
    result_reader = csv.reader(result_FILE, delimiter='\t')
    for result in result_reader:
        probs.append(float(result[0]))
        true_labels.append(int(result[1]=='P'))

f1s = []
this_probs = copy.deepcopy(probs)
for prob in probs:

    pred_labels = [int(this_prob > prob) for this_prob in this_probs]
    f1 = calc_f1(pred_labels, true_labels)
    f1s.append(f1)


print(max(f1s))
