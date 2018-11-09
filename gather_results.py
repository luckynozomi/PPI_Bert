import os

data_name = "BioInfer"
split_method = "SEP_SENT"
append_method = "ORIG"
num_folds = 10
model_name = "trained_model"


def parse_result_line(line):
    name, value = line.split('=')
    name = name.strip()
    value = float(value)
    return name, value


def gather_results():
    dirs = [os.path.join(data_name, split_method, append_method, "fold_"+str(fold)) for fold in range(num_folds)]

    result_dict = {}
    for dir in dirs:
        with open(os.path.join(dir, model_name, "eval_results.txt"), "r") as result_file:
            precision = recall = 0
            for line in result_file:
                name, value = parse_result_line(line)
                if name not in result_dict:
                    result_dict[name] = [value]
                else:
                    result_dict[name].append(value)
                if name == "precision":
                    precision = value
                elif name == "recall":
                    recall = value
            f1 = 2 * precision * recall / (precision + recall)
            print(precision, '\t', recall, '\t', f1)
            if "f1" not in result_dict:
                result_dict["f1"] = [f1]
            else:
                result_dict["f1"].append(f1)

    for name, values in result_dict.items():
        avg = sum(values) / len(values)
        print(name, " has average of ", avg)


if __name__ == "__main__":
    gather_results()
