# backend/scripts/generate_data.py
import csv, random
from paraphrase import paraphrase_text  # we'll create this

positive = [ ... ]  # load from files or in-code arrays
negative = [ ... ]
neutral = [ ... ]

def rule_augment(sentence):
    # simple synonym replacement or add suffix
    # implement lightweight rules to preserve mixed-language forms
    return sentence  # placeholder

def generate_augmented(out_path="data/augmented_dataset.csv", per_seed=5):
    rows = []
    for txt in positive:
        variants = [rule_augment(txt)]
        # call paraphraser to get AI paraphrases
        paras = paraphrase_text(txt, n=per_seed)
        variants += paras
        for v in variants:
            rows.append((v, "positive"))
    # similarly for negative and neutral
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["text","label"])
        for t,l in rows:
            writer.writerow([t,l])
