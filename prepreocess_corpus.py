from lxml import etree as ET
import copy
import random
import os


DATA_PATH = {
    #"PICAD": "PPICorpora/picad_processed.xml",
    "AImed": "PPICorpora/AImed.xml"
}

PCT_TRAIN = .9
NUM_FOLDS = 10
RAND_SEED = 25252


def get_xml_strings(xml_name):

    sent_iter = ET.iterparse(xml_name, tag="sentence")

    ret = []

    for _, sent in sent_iter:

        entities = sent.findall("entity")
        if len(entities) < 2:
            print("Skipped", sent.attrib["text"], "\nwith id", sent.attrib["id"])
            sent.clear()
            continue

        this_sentence = sent.attrib["text"]

        entity_triplet_list = []

        for this_entity in sent.findall("entity"):
            id = this_entity.attrib["id"]
            char_offset = this_entity.attrib["charOffset"].split(',')[0]
            start = int(char_offset.split('-')[0])
            end = int(char_offset.split('-')[1]) + 1
            entity_triplet_list.append((start, end, id))

        entity_triplet_list.sort(key=lambda elem: elem[0])

        segmented_sent = []
        entity_pos_dict = {}
        prev_end = 0
        entity_pos = 0
        for start, end, elem in entity_triplet_list:
            if start == prev_end:
                segmented_sent.append(this_sentence[start:end].strip())
                entity_pos_dict[elem] = entity_pos
                entity_pos += 1
            else:
                segmented_sent.append(this_sentence[prev_end:start].strip())
                segmented_sent.append(this_sentence[start:end].strip())
                entity_pos_dict[elem] = entity_pos + 1
                entity_pos += 2
            prev_end = end
        segmented_sent.append(this_sentence[prev_end:].strip())

        for this_pair in sent.findall("pair"):

            this_label = (this_pair.attrib["interaction"] == "True")

            this_seg_sent = copy.deepcopy(segmented_sent)
            entity1 = this_pair.attrib["e1"]
            entity2 = this_pair.attrib["e2"]
            for entity, pos in entity_pos_dict.items():
                if entity == entity1:
                    protA_name = this_seg_sent[pos]
                    this_seg_sent[pos] = "PROTEIN1"
                elif entity == entity2:
                    protB_name = this_seg_sent[pos]
                    this_seg_sent[pos] = "PROTEIN2"
                else:
                    this_seg_sent[pos] = "PROTEIN"
            this_sent = ' '.join(this_seg_sent)

            if this_label is True:
                ret.append(sent.attrib["id"] + '\t' + "P" + '\t' + this_sent)
            else:
                ret.append(sent.attrib["id"] + '\t' + "N" + '\t' + this_sent)

    return ret


def flatten_strings(strings):
    return [instance for sublist in strings for instance in sublist]


def split_strings(strings, percent_train, split_method):
    random.seed(RAND_SEED)
    ret_train = []
    ret_dev = []

    if split_method == "SEP_SENT":
        num_sents = len(strings)
        num_train = int(num_sents * percent_train)
        random.shuffle(strings)
        for idx in range(0, num_train):
            for instance in strings[idx]:
                ret_train.append(instance)
        for idx in range(num_train, num_sents):
            for instance in strings[idx]:
                ret_dev.append(instance)
    else:
        flat_strings = flatten_strings(strings)
        num_instances = len(flat_strings)
        num_train = int(num_instances * percent_train)
        random.shuffle(flat_strings)
        ret_train = flat_strings[0:num_train]
        ret_dev = flat_strings[num_train:]

    return ret_train, ret_dev


def append(data_list, append_method):
    if append_method == "ORIG":
        return data_list
    else:
        pos_list = []
        neg_list = []
        for instance in data_list:
            if instance[0] == 'P':
                pos_list.append(instance)
            else:
                neg_list.append(instance)
        num_pos = len(pos_list)
        num_neg = len(neg_list)

        if num_neg < num_pos:
            raise NotImplementedError

        num_folds = int(num_neg / num_pos - 1)
        for _ in range(num_folds):
            pos_list += copy.deepcopy(pos_list)
        ret_list = pos_list + neg_list
        random.shuffle(ret_list)
    return ret_list


def save_file(data_list, file_name):
    with open(file_name, "w") as f:
        for instance in data_list:
            f.write(instance + '\n')


if __name__ == "__main__":

    out_dir = "processed_corpus"
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    for data_name, data_path in DATA_PATH.items():

        ret = get_xml_strings(data_path)

        save_file(ret, os.path.join(out_dir, data_name+".tsv"))
