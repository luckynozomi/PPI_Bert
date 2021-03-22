import csv
import copy
import os


result_path = "/home/manbish/Documents/PPI_Bert/AImed_Results"
result_files = [os.path.join(result_path, str(i)+"test_results.tsv") for i in range(10)]


def calc_f1(pred_labels, true_labels):
    t_p = sum([pred_labels[i] * true_labels[i] for i in range(len(pred_labels))])
    true = sum(true_labels)
    positive = sum(pred_labels)
    try:
        precision = t_p / positive
        recall = t_p / true
        f_1 = precision * recall * 2 / (precision + recall)
        return precision, recall, f_1
    except ZeroDivisionError:
        return 0, 0, 0


def read_result_file(result_file):
    probs = []
    true_labels = []
    with open(result_file, "r") as result_FILE:
        result_reader = csv.reader(result_FILE, delimiter='\t')
        for result in result_reader:
            probs.append(float(result[0]))
            true_labels.append(int(result[1]=='P'))
    return probs, true_labels


def calc_average_f1(all_results, this_cutoff, number_of_results):
    precisions = []
    recalls = []
    f1s = []

    for probs, true_labels in all_results:
        pred_labels = [int(prob >= this_cutoff) for prob in probs]
        precision, recall, f1 = calc_f1(pred_labels, true_labels)
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return sum(precisions)/number_of_results, sum(recalls)/number_of_results, sum(f1s)/number_of_results


if __name__ == "__main__":
    all_probs = []
    all_results = []
    stats = []
    for result_file in result_files:
        probs, true_labels = read_result_file(result_file)
        all_probs += probs
        all_results.append([probs, true_labels])

    all_probs = list(set(all_probs))
    all_probs.sort()
    for this_cutoff in all_probs:
        this_stat = calc_average_f1(all_results, this_cutoff, len(result_files))
        stats.append(list(this_stat))

    stats.sort(key=lambda i: i[2], reverse=True)
    print(stats[0])
