import os
TEST_DIR = "AImed_CC_Results"
import csv


def get_results(file_path):
    positive_prob = []
    real_label = []
    with open(file_path, "r") as file:
        file_reader = csv.reader(file, delimiter='\t')
        for line in file_reader:
            positive_prob.append(float(line[0]))
            real_label.append(int(line[1]=="P"))
    return positive_prob, real_label


def make_prediction(positive_probs, vote_to_win):
    ret = []
    for i in range(len(positive_probs[0])):
        probs = [positive_prob[i] for positive_prob in positive_probs]
        votes = sum([prob >= 0.5 for prob in probs])
        if votes >= vote_to_win:
            ret.append(1)
        else:
            ret.append(0)
    return ret


def calc_pr(pred_label, real_label):
    t_p = sum([pred_label[i] * real_label[i] for i in range(len(pred_label))])
    true = sum(real_label)
    positive = sum(pred_label)
    precision = t_p / positive
    recall = t_p / true
    f_1 = 2 * precision * recall / (precision + recall)
    print("precision", precision)
    print("recall", recall)
    print("F1", f_1)


def gather_test_files():
    positive_probs = []
    for file_name in os.listdir(TEST_DIR):
        if "test_results" not in file_name:
            continue
        positive_prob, real_label = get_results(os.path.join(TEST_DIR, file_name))
        positive_probs.append(positive_prob)

    pred_label = make_prediction(positive_probs, vote_to_win=5)
    calc_pr(pred_label, real_label)


if __name__ =="__main__":
    gather_test_files()

