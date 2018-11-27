from lxml import etree as ET
import itertools


PICAD_PATH="/Users/xinsui/Dropbox/PPI_Bert/PPICorpora/picad.xml"


def find_offsets(entity_name, text):
    ret = []
    start = text.find(entity_name)
    if start == -1:
        raise ValueError
    while start != -1:
        end = start + len(entity_name)
        ret.append([start, end])
        start = text.find(entity_name, end)
    return ret


def is_subrange(subrange, range):
    if subrange[0] >= range[0] and subrange[1] <= range[1]:
        return True
    else:
        return False


def find_positions(entities, text):
    ret = {}
    for entity in entities:
        entity_name = entity.attrib["name"]
        try:
            entity_offsets = find_offsets(entity_name, text)
        except ValueError:
            print("Cannot parse " + text + " with entity " + entity_name)
            continue
        for index, char_offset in enumerate(entity_offsets):
            if entity_name not in ret.keys():
                ret[entity_name] = [char_offset]
            else:
                ret[entity_name].append(char_offset)

    for entity_1, entity_2 in itertools.permutations(ret.keys(), 2):
        if entity_1 in entity_2:
            for offset in ret[entity_1]:
                if any([is_subrange(offset, entity_2_offset) for entity_2_offset in ret[entity_2]]) is True:
                    ret[entity_1].remove(offset)
    real_ret = {}
    for entity in ret.keys():
        if len(ret[entity]) > 0:
            real_ret[entity] = ret[entity]
    return real_ret


def find_word_position(text, start_position, end_position):
    return text[0:start_position].count(' ') + 1


def build_pos_to_entity_dict(entities_poses, entities, sent_id, text):
    ret = {}
    entity_index = 0
    for entity_name, poses in entities_poses.items():
        for start_position, end_position in poses:

            pos = find_word_position(text, start_position, end_position)
            entity = None
            for this_entity in entities:
                if this_entity.attrib["name"] == entity_name:
                    entity = this_entity

            this_element = ET.Element(
                "entity",
                id=sent_id+".e"+str(entity_index),
                text=entity_name,
                charOffset=str(start_position)+'-'+str(end_position-1),
                type=entity.attrib["type"]
            )

            entity_index += 1
            ret[pos] = this_element
    return ret


def find_interaction(pos1, pos2, interactions):
    set_poses = set([pos1, pos2])
    for interaction in interactions:
        pos1 = int(interaction.attrib["pos1"])
        pos2 = int(interaction.attrib["pos2"])
        if set([pos1, pos2]) == set_poses:
            return interaction
    return None


sent_iter = ET.iterparse(PICAD_PATH, tag="sentence")

out = ET.Element("corpus", source="PICAD")

for _, sentence in sent_iter:
    sent_id = sentence.attrib["id"]
    text = sentence.attrib["text"]
    entities = sentence.findall("entity")
    entities_poses = find_positions(entities, text)
    pos_to_entity_dict = build_pos_to_entity_dict(entities_poses, entities, sent_id, text)

    pairs = []
    pair_id = 0
    interactions = sentence.findall("interaction")
    for (pos1, entity1), (pos2, entity2) in itertools.combinations(pos_to_entity_dict.items(), 2):

        interaction = find_interaction(pos1, pos2, interactions)
        if interaction is None:
            this_pair = ET.Element(
                "pair",
                e1=entity1.attrib["id"],
                e2=entity2.attrib["id"],
                id=sent_id+".p"+str(pair_id),
                interaction="False",
                iwpos="None",
                iw="None",
                dir="None"
            )
        else:
            this_pair = ET.Element(
                "pair",
                e1=entity1.attrib["id"],
                e2=entity2.attrib["id"],
                id=sent_id+".p"+str(pair_id),
                interaction="True",
                iwpos=interaction.attrib["iwpos"],
                iw=interaction.attrib["iw"],
                dir=interaction.attrib["dir"]
            )
        pairs.append(this_pair)
        pair_id += 1

    out_sentence = ET.Element("sentence", id=sent_id, text=text)
    for entity in pos_to_entity_dict.values():
        out_sentence.append(entity)
    for pair in pairs:
        out_sentence.append(pair)

    out.append(out_sentence)

ET.dump(out)