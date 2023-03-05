from copy import deepcopy

import nltk
import pandas as pd
import spacy
from datasets import ClassLabel, Dataset, concatenate_datasets
from pattern import en
from tqdm import tqdm


def lower_first(s):
    return s[0].lower() + s[1:]


def upper_first(s):
    return s[0].upper() + s[1:]


class nli_augmentations:
    """Credit: @Aatlantise Link: https://github.com/Aatlantise/syntactic-augmentation-nli"""

    def __init__(self) -> None:
        self.nlp = spacy.load("en_core_web_sm")
        self.ner = self.nlp.create_pipe("ner")
        self.parser = self.nlp.create_pipe("parser")
        self.lemmatizer = nltk.stem.WordNetLemmatizer()

        self.present_to_past = {}
        self.present_to_vb = {}
        self.present_to_vbz = {}

    def get_np_head(self, np):
        head = None
        if np.label() == "NP" and np[0].label() == "DT":
            head_candidates = [x for x in np[1:] if x.label().startswith("NN")]
            if len(head_candidates) == 1:
                # > 1: Complex noun phrases unlikely to be useful
                # 0: Pronominal subjects like "many"
                head = self.lemmatizer.lemmatize(head_candidates[0][0])
        return head

    def get_np_number(self, np):
        number = None
        if np[0].label() == "NP":
            np = np[0]
        head_candidates = [x for x in np if x.label().startswith("NN")]
        if len(head_candidates) == 1:
            label = head_candidates[0].label()
            number = en.PLURAL if label == "NNS" else en.SINGULAR
        elif len(head_candidates) > 1:
            number = en.PLURAL
        return number

    def get_vp_head(self, vp):
        head = None
        if vp.label() == "VP":
            while True:
                nested_vps = [x for x in vp[1:] if x.label() == "VP"]
                if len(nested_vps) == 0:
                    break
                vp = nested_vps[0]
            if vp[0].label().startswith("VB"):
                head = vp[0][0].lower()

        return (head, vp[0].label())

    def passivize_vp(self, vp, subj_num=en.SINGULAR):
        head = None
        flat = vp.flatten()
        if vp.label() == "VP":
            nesters = []
            while True:
                nesters.append(vp[0][0])
                nested_vps = [x for x in vp[1:] if x.label() == "VP"]
                if len(nested_vps) == 0:
                    break
                vp = nested_vps[0]
            label = vp[0].label()
            if label.startswith("VB"):
                head = vp[0][0].lower()
                if len(nesters) > 1:
                    passivizer = "be"
                elif label in ["VBP", "VB", "VBZ"]:
                    # 'VB' here (not nested) is a POS tag error
                    passivizer = "are" if subj_num == en.PLURAL else "is"
                elif label == "VBD" or label == "VBN":
                    # 'VBN' here (not nested) is a POS tag error
                    passivizer = "were" if subj_num == en.PLURAL else "was"
                    # Alternatively, figure out the number of the subject
                    # to decide whether it's was or were
                else:
                    passivizer = "is"
                vbn = en.conjugate(head, "ppart")

        return f"{passivizer} {vbn} by"

    def infer(self, dataset, n_workers="max"):
        datacols = list(dataset.features.keys())
        w_inv_orig = {k: [] for k in datacols}
        w_inv_trsf = {k: [] for k in datacols}
        w_pass_orig = {k: [] for k in datacols}
        w_pass_trsf = {k: [] for k in datacols}

        for i in tqdm(range(len(dataset))):
            tree = nltk.tree.Tree.fromstring(dataset["hypothesis_parse"][i])
            ss = [x for x in tree.subtrees() if x.label() == "S"]

            for s in ss[:1]:
                if len(s) < 2:  # Not a full NP + VP sentence
                    continue

                subj_head = self.get_np_head(s[0])
                if subj_head is None:
                    continue
                subject_number = self.get_np_number(s[0])

                k = 1

                while (s[k].label() not in ("VP", "SBAR", "ADJP")) and (k < len(s) - 1):
                    k += 1

                if k == len(s) - 1:
                    continue

                vp_head = self.get_vp_head(s[k])

                if vp_head[0] is None:
                    continue

                subj = " ".join(s[0].flatten())
                arguments = tuple(x.label() for x in s[1][1:])

                # TODO: resolve this try/except condition
                flag_except = False
                try:
                    if arguments != ("NP",) or en.lemma(vp_head[0]) in ["be", "have"]:
                        continue
                except Exception:
                    flag_except = True

                if flag_except:
                    continue

                direct_object = " ".join(s[1][1].flatten())

                object_number = self.get_np_number(s[1][1])

                if object_number is None:
                    # Personal pronoun, very complex NP, or parse error
                    continue

                lookup = en.tenses(vp_head[0])

                if len(lookup) == 0:
                    if vp_head[0][-2:]:
                        tense = en.PAST
                    else:
                        tense = en.PRESENT
                else:
                    if en.tenses(vp_head[0])[0][0] == "past":
                        tense = en.PAST
                    else:
                        tense = en.PRESENT

                subjobj_rev_hyp = (
                    " ".join(
                        [
                            upper_first(direct_object),
                            # keep tense
                            en.conjugate(vp_head[0], number=object_number, tense=tense),
                            lower_first(subj),
                        ]
                    )
                    + "."
                )

                passive_hyp_same_meaning = (
                    " ".join(
                        [
                            upper_first(direct_object),
                            self.passivize_vp(s[k], object_number),
                            lower_first(subj),
                        ]
                    )
                    + "."
                )

                passive_hyp_inverted = (
                    " ".join(
                        [subj, self.passivize_vp(s[k], subject_number), direct_object]
                    )
                    + "."
                )

                if dataset["label"][i] == 0:  # entailed
                    w_inv_orig["premise"].append(dataset["premise"][i])
                    w_inv_orig["hypothesis"].append(subjobj_rev_hyp)
                    w_inv_orig["label"].append(1)
                    w_inv_orig["mapping"].append(i)

                    for k in datacols:
                        if k not in ["premise", "hypothesis", "label", "mapping"]:
                            w_inv_orig[k].append(dataset[k][i])

                w_inv_trsf["premise"].append(dataset["hypothesis"][i])
                w_inv_trsf["hypothesis"].append(subjobj_rev_hyp)
                w_inv_trsf["label"].append(1)
                w_inv_trsf["mapping"].append(i)
                for k in datacols:
                    if k not in ["premise", "hypothesis", "label", "mapping"]:
                        w_inv_trsf[k].append(dataset[k][i])

                w_pass_orig["premise"].append(dataset["premise"][i])
                w_pass_orig["hypothesis"].append(passive_hyp_same_meaning)
                w_pass_orig["label"].append(dataset["label"][i])
                w_pass_orig["mapping"].append(i)
                for k in datacols:
                    if k not in ["premise", "hypothesis", "label", "mapping"]:
                        w_pass_orig[k].append(dataset[k][i])

                w_pass_trsf["premise"].append(dataset["hypothesis"][i])
                w_pass_trsf["hypothesis"].append(passive_hyp_inverted)
                w_pass_trsf["label"].append(1)
                w_pass_trsf["mapping"].append(i)
                for k in datacols:
                    if k not in ["premise", "hypothesis", "label", "mapping"]:
                        w_pass_trsf[k].append(dataset[k][i])

                w_pass_trsf["premise"].append(dataset["hypothesis"][i])
                w_pass_trsf["hypothesis"].append(passive_hyp_same_meaning)
                w_pass_trsf["label"].append(0)
                w_pass_trsf["mapping"].append(i)
                for k in datacols:
                    if k not in ["premise", "hypothesis", "label", "mapping"]:
                        w_pass_trsf[k].append(dataset[k][i])

        return concatenate_datasets(
            [
                Dataset.from_pandas(pd.DataFrame(w_inv_orig)),
                Dataset.from_pandas(pd.DataFrame(w_inv_trsf)),
                Dataset.from_pandas(pd.DataFrame(w_pass_orig)),
                Dataset.from_pandas(pd.DataFrame(w_pass_trsf)),
            ]
        ).cast_column(
            "label",
            ClassLabel(num_classes=3, names=["entailment", "neutral", "contradiction"]),
        )
