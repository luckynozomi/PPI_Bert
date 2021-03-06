import os

data_name = "AImed"
split_method = "SEP_INS"
append_method = "ORIG"
num_folds = 10
num_iters = 10
model_name = "Instance_Model"


def parse_result_line(line):
    name, value = line.split('=')
    name = name.strip()
    value = float(value)
    return name, value


def gather_files():
    ret = []

    dirs = [os.path.join(data_name, append_method, "fold_"+str(fold)) for fold in range(num_folds)]

    for dir in dirs:
        ret.append(os.path.join(dir, model_name, "eval_results.txt"))

    return ret


def gather_CC_files():
    ret = []
    dirs = [os.path.join(data_name, append_method, model_name, "iter-"+str(iteration)) for iteration in range(num_iters)]
    for dir in dirs:
        ret.append(os.path.join(dir, "eval_results.txt"))
    return ret


def gather_results():
    dirs = [os.path.join(data_name, split_method, append_method, "fold_"+str(fold)) for fold in range(num_folds)]
    
    result_dict = {}
    for ret_file in gather_files():
        with open(ret_file, "r") as result_file:
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
