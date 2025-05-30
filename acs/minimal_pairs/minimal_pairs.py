import pickle
import random
import re
from collections import Counter, defaultdict

from nltk.tokenize.treebank import TreebankWordDetokenizer
from tqdm import tqdm


class CSMinimalPairs:

    def __init__(self, remove_chinese_space):
        self.already_swapped = defaultdict(int)
        self.dt = TreebankWordDetokenizer()
        self.reasons = []
        self.remove_chinese_space = remove_chinese_space

        self.all_resulting_minimal_pairs = []

    def make_minimal_pair(self, data):

        options = self.find_options(data)

        random.shuffle(options)

        options = [self.make_swap(data, pivot, change) for pivot, change in options]

        options = sorted(
            options,
            key=lambda words_labels_change: self.already_swapped.get(
                words_labels_change[2], 0
            ),
        )

        while len(options) > 0:
            words_mp, labels_mp, pivot, change_pair, change_pair_idxs = options.pop(0)

            # # words_mp, labels_mp, change_pair = self.make_swap(data, pivot, change)

            if self.already_swapped[change_pair] == 1:
                self.reasons.append("Already swapped")
                continue

            if not ("lang2" in labels_mp and "lang1" in labels_mp):
                self.reasons.append("lang1 and lang2 not both in labels mp")
                continue

            text_original = self.dt.detokenize(data["tokens"])
            text_mp = self.dt.detokenize(words_mp)

            chinese_regex = r"[\u4e00-\u9fff]+"

            if self.remove_chinese_space:
                for i in range(len(text_original) - 2, 0, -1):
                    if (
                        text_original[i] == " "
                        and re.findall(chinese_regex, text_original[i - 1])
                        and re.findall(chinese_regex, text_original[i + 1])
                    ):
                        text_original = text_original[:i] + text_original[i + 1 :]

                for i in range(len(text_mp) - 2, 0, -1):
                    if (
                        text_mp[i] == " "
                        and re.findall(chinese_regex, text_mp[i - 1])
                        and re.findall(chinese_regex, text_mp[i + 1])
                    ):
                        text_mp = text_mp[:i] + text_mp[i + 1 :]

            # check we definitely changed something!
            if (
                text_mp.lower().replace(" ", "")
                == data["text"].lower().replace(" ", "")
                or text_mp.lower().replace(" ", "")
                == data["text_lang2"].lower().replace(" ", "")
                or text_mp.lower().replace(" ", "")
                == data["text_lang1"].lower().replace(" ", "")
            ):
                self.reasons.append("Identical sentence")
                continue

            # make sure change words are alphanumeric

            if (
                not change_pair[1].replace(" ", "").isalpha()
                or not change_pair[0].replace(" ", "").isalpha()
            ):
                self.reasons.append("Change word not alphanumeric")
                continue

            # check we haven't changed the number of islands
            c = Counter(data["langs"])

            assert c["lang2"] > 0
            assert c["lang1"] > 0
            if c["lang1"] > c["lang2"]:
                old_islands = self.count_num_islands(data["langs"], "lang2")
                new_islands = self.count_num_islands(labels_mp, "lang2")
                if old_islands != new_islands:
                    self.reasons.append("Changed number of islands")
                    continue
            else:
                old_islands = self.count_num_islands(data["langs"], "lang1")
                new_islands = self.count_num_islands(labels_mp, "lang1")
                if old_islands != new_islands:
                    self.reasons.append("Changed number of islands")
                    continue

            self.already_swapped[change_pair] += 1

            assert len(words_mp) == len(labels_mp)

            data["real_sentence"] = text_original
            data["manipulated_sentence"] = text_mp
            data["manipulated_tokens"] = words_mp
            data["manipulated_langs"] = labels_mp

            data["change_pair"] = change_pair
            data["change_pair_idxs"] = change_pair_idxs
            data["pivot"] = pivot

            self.all_resulting_minimal_pairs.append(data)

        return None

    def count_num_islands(self, labels, lang):

        count = 0
        curr_lang = None

        for idx in range(len(labels)):
            if labels[idx] == lang:
                if curr_lang is None:
                    curr_lang = lang
            else:
                if curr_lang == lang:
                    count += 1
                    curr_lang = None

        if curr_lang == lang:
            count += 1

        return count

    def make_swap(self, data, pivot, change):

        words_mp = data["tokens"].copy()
        labels_mp = data["langs"].copy()

        change_from_lang = data["langs"][change]

        if change_from_lang == "lang1":

            alignments_to_monolingual = data["alignments_cs_lang2"]
            alignments_to_cs = data["alignments_lang2_cs"]
            dependencies_monolingual = data["dependencies_lang2"]
            words_monolingual = data["words_lang2"]
            change_to_lang = "lang2"
        elif change_from_lang == "lang2":

            alignments_to_monolingual = data["alignments_cs_lang1"]
            alignments_to_cs = data["alignments_lang1_cs"]
            dependencies_monolingual = data["dependencies_lang1"]
            words_monolingual = data["words_lang1"]
            change_to_lang = "lang1"
        else:
            raise Exception

        all_translated_change_idxs = []

        for i, en_change_word_idx in enumerate(
            sorted(alignments_to_monolingual[change])
        ):

            translated_change_idxs = self.compute_translated_change(
                en_change_word_idx,
                alignments_to_cs,
                dependencies_monolingual,
            )

            all_translated_change_idxs.extend(translated_change_idxs)

        for monolingual_idx in all_translated_change_idxs:
            for aligned_cs_idx in alignments_to_cs[monolingual_idx]:
                if aligned_cs_idx not in [pivot, change]:
                    words_mp[aligned_cs_idx] = ""

        all_translated_change = [
            words_monolingual[idx] for idx in sorted(all_translated_change_idxs)
        ]

        words_mp = words_mp[:change] + all_translated_change + words_mp[change + 1 :]

        labels_mp = (
            labels_mp[:change]
            + [change_to_lang] * len(all_translated_change)
            + labels_mp[change + 1 :]
        )

        # remove indexes in words_mp and labels_mp where words are empty

        labels_mp = [label for label, word in zip(labels_mp, words_mp) if word != ""]
        words_mp = [word for word in words_mp if word != ""]

        return (
            words_mp,
            labels_mp,
            pivot,
            (data["tokens"][change], " ".join(all_translated_change)),
            (change, all_translated_change_idxs),
        )

    def compute_translated_change(
        self,
        monolingual_change_word_idx,
        alignments_to_cs,
        dependencies_monolingual,
    ):

        below_mono_idx = monolingual_change_word_idx - 1
        translated_below_words_idxs = []

        while (
            below_mono_idx >= 0
            and len(translated_below_words_idxs)
            >= monolingual_change_word_idx - below_mono_idx - 1
        ):
            for head, deprel, dep in dependencies_monolingual:
                if (
                    head == monolingual_change_word_idx
                    and dep == below_mono_idx
                    and len(alignments_to_cs[dep]) == 0
                ):
                    translated_below_words_idxs.insert(0, dep)
                elif (
                    head == below_mono_idx
                    and dep == monolingual_change_word_idx
                    and len(alignments_to_cs[head]) == 0
                ):
                    translated_below_words_idxs.insert(0, head)

                    break

            below_mono_idx -= 1

        translated_change_word_idx = monolingual_change_word_idx

        above_mono_idx = monolingual_change_word_idx + 1
        translated_above_words_idxs = []

        while (
            above_mono_idx in alignments_to_cs
            and len(translated_above_words_idxs)
            >= above_mono_idx - monolingual_change_word_idx - 1
        ):
            for head, deprel, dep in dependencies_monolingual:
                if (
                    head == monolingual_change_word_idx
                    and dep == above_mono_idx
                    and len(alignments_to_cs[dep]) == 0
                ):
                    translated_above_words_idxs.append(dep)
                elif (
                    head == above_mono_idx
                    and dep == monolingual_change_word_idx
                    and len(alignments_to_cs[head]) == 0
                ):
                    translated_above_words_idxs.append(head)

                    break

            above_mono_idx += 1

        translated_change_idxs = (
            translated_below_words_idxs
            + [translated_change_word_idx]
            + translated_above_words_idxs
        )

        return translated_change_idxs

    def find_options(self, data):

        options = []

        for idx, label in enumerate(data["langs"]):

            if idx == 0:
                continue
            elif idx == len(data["langs"]) - 1:
                continue
            elif idx == 1:
                local_options = [(idx, idx + 1)]
            elif idx == len(data["langs"]) - 2:
                local_options = [(idx, idx - 1)]
            else:
                local_options = [(idx, idx + 1), (idx, idx - 1)]

            # Check if there is even a language switch here, i.e. one of them is "lang1" and the other is "lang2"
            for o in local_options.copy():
                if not sorted(
                    list(set([data["langs"][o[0]], data["langs"][o[1]]]))
                ) == ["lang1", "lang2"]:
                    local_options.remove(o)

            for o in local_options.copy():
                if o[1] == o[0] + 1:
                    if not data["langs"][o[0] - 1] == data["langs"][o[0]]:
                        local_options.remove(o)
                        self.reasons.append("Single word island")
                elif o[1] == o[0] - 1:
                    if not data["langs"][o[0] + 1] == data["langs"][o[0]]:
                        local_options.remove(o)
                        self.reasons.append("Single word island")

            # Check the change word is not in a single word island
            for o in local_options.copy():
                if o[1] == o[0] + 1:
                    if not data["langs"][o[1] + 1] == data["langs"][o[1]]:
                        local_options.remove(o)
                        self.reasons.append("Single word island")
                elif o[1] == o[0] - 1:
                    if not data["langs"][o[1] - 1] == data["langs"][o[1]]:
                        local_options.remove(o)
                        self.reasons.append("Single word island")

            # Check if there is MWE preventing the swap

            for o in local_options.copy():
                if not (data["mwe_bio"][o[1]] == "O" and data["mwe_bio"][o[0]] == "O"):
                    local_options.remove(o)
                    self.reasons.append("MWE")

            # Check if there is a NOUN preventing the swap

            for o in local_options.copy():

                for aligned_idx in data["alignments_cs_lang1"][o[0]]:
                    if (
                        data["upos_lang1"][aligned_idx] in ["NOUN", "PROPN"]
                        and o in local_options
                    ):
                        local_options.remove(o)
                        self.reasons.append("NOUN")
                        break

                for aligned_idx in data["alignments_cs_lang1"][o[1]]:
                    if (
                        data["upos_lang1"][aligned_idx] in ["NOUN", "PROPN"]
                        and o in local_options
                    ):
                        local_options.remove(o)
                        self.reasons.append("NOUN")
                        break

                for aligned_idx in data["alignments_cs_lang2"][o[0]]:
                    if (
                        data["upos_lang2"][aligned_idx] in ["NOUN", "PROPN"]
                        and o in local_options
                    ):
                        local_options.remove(o)
                        self.reasons.append("NOUN")
                        break

                for aligned_idx in data["alignments_cs_lang2"][o[1]]:
                    if (
                        data["upos_lang2"][aligned_idx] in ["NOUN", "PROPN"]
                        and o in local_options
                    ):
                        local_options.remove(o)
                        self.reasons.append("NOUN")
                        break

            # Check in a phrase with an outward pointing dependency

            for o in local_options.copy():

                num_lang1 = sum(1 for lang in data["langs"] if lang == "lang1")
                num_lang2 = sum(1 for lang in data["langs"] if lang == "lang2")

                if num_lang1 > num_lang2:
                    mostfrequent_lang = "lang1"
                    leastfrequent_lang = "lang2"
                else:
                    mostfrequent_lang = "lang2"
                    leastfrequent_lang = "lang1"

                if data["langs"][o[0]] == mostfrequent_lang:
                    matrix_idx = o[0]
                    embedded_idx = o[1]
                else:
                    matrix_idx = o[1]
                    embedded_idx = o[0]

                below_idx = embedded_idx

                while (
                    data["langs"][below_idx - 1] == data["langs"][embedded_idx]
                    and below_idx - 1 >= 0
                ):
                    below_idx -= 1

                above_idx = embedded_idx + 1

                while (
                    above_idx + 1 <= len(data["langs"]) - 1
                    and data["langs"][above_idx + 1] == data["langs"][embedded_idx]
                ):
                    above_idx += 1

                head_range = range(below_idx, above_idx)

                real_code_switch = False

                for head_in_cs_span in head_range:

                    for aligned_idx in data["alignments_cs_lang1"][head_in_cs_span]:
                        for head, rel, dep in data["dependencies_lang1"]:
                            if head == aligned_idx:
                                for aligned_cs in data["alignments_lang1_cs"][dep]:
                                    if data["langs"][aligned_cs] == mostfrequent_lang:
                                        real_code_switch = True

                for head_in_cs_span in head_range:
                    for aligned_idx in data["alignments_cs_lang2"][head_in_cs_span]:
                        for head, rel, dep in data["dependencies_lang2"]:
                            if head == aligned_idx:
                                for aligned_cs in data["alignments_lang2_cs"][dep]:
                                    if data["langs"][aligned_cs] == mostfrequent_lang:

                                        real_code_switch = True

                if not real_code_switch:
                    local_options.remove(o)
                    self.reasons.append("No outward pointing dependency")

            # check we have associated word to swap:
            for o in local_options.copy():
                if data["langs"][o[1]] == "lang2":
                    if len(data["alignments_cs_lang1"][o[1]]) == 0:
                        local_options.remove(o)
                        self.reasons.append("No aligned word")

                if data["langs"][o[1]] == "lang1":
                    if len(data["alignments_cs_lang2"][o[1]]) == 0:
                        local_options.remove(o)
                        self.reasons.append("No aligned word")

            # check the alignments for the change word and the pivot word do not overlap
            for o in local_options.copy():
                if data["langs"][o[1]] == "lang2":
                    pivot_alignments = set(data["alignments_cs_lang1"][o[0]])
                    change_alignments = set(data["alignments_cs_lang1"][o[1]])
                    if len(pivot_alignments.intersection(change_alignments)) > 0:
                        local_options.remove(o)
                        self.reasons.append("Overlapping alignments")

                if data["langs"][o[1]] == "lang1":
                    pivot_alignments = set(data["alignments_cs_lang2"][o[0]])
                    change_alignments = set(data["alignments_cs_lang2"][o[1]])
                    if len(pivot_alignments.intersection(change_alignments)) > 0:
                        local_options.remove(o)
                        self.reasons.append("Overlapping alignments")

            # check if the swapped word is actualy different to the original word
            for o in local_options.copy():
                if data["langs"][o[1]] == "lang2":
                    if len(data["alignments_cs_lang1"][o[1]]) > 1:
                        continue
                    elif (
                        data["words_lang1"][
                            data["alignments_cs_lang1"][o[1]][0]
                        ].lower()
                        == data["tokens"][o[1]].lower()
                    ):
                        local_options.remove(o)
                        self.reasons.append("Identical aligned word")

                if data["langs"][o[1]] == "lang1":
                    if len(data["alignments_cs_lang2"][o[1]]) > 1:
                        continue
                    elif (
                        data["words_lang2"][
                            data["alignments_cs_lang2"][o[1]][0]
                        ].lower()
                        == data["tokens"][o[1]].lower()
                    ):

                        local_options.remove(o)
                        self.reasons.append("Identical aligned word")

            options.extend(local_options)

        return options


def make_minimal_pairs(
    processed_data_file="data/cache/processed_data.pkl", remove_chinese_space=False
):

    processed_data = pickle.load(open(processed_data_file, "rb"))

    CSMP = CSMinimalPairs(remove_chinese_space=remove_chinese_space)

    for data in tqdm(processed_data, desc="Making minimal pairs"):
        CSMP.make_minimal_pair(data)

    output_data = CSMP.all_resulting_minimal_pairs
    # random.shuffle(output_data) # we shuffled here
    output_data = {data["id"]: data for data in output_data}

    return output_data


if __name__ == "__main__":

    results = make_minimal_pairs()

    for id, data in results.items():
        print(f"ID: {id}")
        print(f" {data['real_sentence']}")
        print(f"*{data['manipulated_sentence']}")
