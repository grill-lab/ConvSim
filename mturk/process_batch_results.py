import argparse
from ast import literal_eval
from collections import Counter

import pandas as pd
from IPython import embed


def main(args):
    #input_csv = pd.read_csv(args.input_csv) # not needed?
    batch = pd.read_csv(args.batch_results)

    ret = []
    bad_workers, good_workers = [], []
    incorrect, correct = 0, 0 # for test cases
    for i, row in batch.iterrows():
        try:
            anns = literal_eval(row["Answer.relevance_assessments"])
        except:
            print("literal_eval error:", row["Answer.relevance_assessments"])
            continue
        for ann in anns[1:]:
            btn = ann[0]
            judgement = btn[-3]
            btn_i = btn[-1]
            # if it's a test case
            if f"Input.correct_{btn_i}" in batch.columns and type(row[f"Input.correct_{btn_i}"]) == str:
                test_correct = row[f"Input.correct_{btn_i}"]
                if test_correct[-1] != judgement[-1]:
                    # failed a test case:
                    print(row["WorkerId"], test_correct, judgement, 
                          row[f"Input.facet_desc_{btn_i}"], "|",
                          row[f"Input.question_{btn_i}"], "|",
                          row[f"Input.answer_1_{btn_i}"], "|",
                          row[f"Input.answer_2_{btn_i}"])
                    bad_workers.append(row["WorkerId"])
                    # put in a rejection to all worker ids that failed?
                    incorrect += 1
                else:
                    good_workers.append(row["WorkerId"])
                    correct += 1
            else: # it's not a test case
                is_shuffled = row[f"Input.shuffled_{btn_i}"]
                if judgement == "1" and not is_shuffled or judgement == "2" and is_shuffled:
                    vote = "gpt3"
                else:
                    vote = "gpt2"
                ret.append({
                    "WorkerId": row["WorkerId"],
                    "facet_desc": row[f"Input.facet_desc_{btn_i}"],
                    "question": row[f"Input.question_{btn_i}"],
                    "answer_1": row[f"Input.answer_1_{btn_i}"] if not is_shuffled else row[f"Input.answer_2_{btn_i}"],
                    "answer_2": row[f"Input.answer_2_{btn_i}"] if not is_shuffled else row[f"Input.answer_1_{btn_i}"],
                    "annotation": vote,
                    })
    
    bad = Counter(bad_workers)
    good = Counter(good_workers)
    for k, v in bad.items():
        print(k, v, "|", v / (v+good.get(k, 0))*100, "%")
    print(f"Correct test cases: {correct}")
    print(f"Missed test cases: {incorrect}")
    df = pd.DataFrame(ret)
    print(df.annotation.value_counts())
    
    grp = df.groupby(["facet_desc", "question", "answer_1", "answer_2"]).agg(list)
    print(grp["annotation"].value_counts())

    really_bad_ones = []
    for k, v in bad.items():
        if v > 2 and good.get(k, 0) <= 1:
            really_bad_ones.append(k)
            print(k, v, "|", v / (v+good.get(k, 0))*100, "%")
    print(f"Rejected: {len(really_bad_ones)}.")
    # reject users
    rejection_text = "Reckless submission; failed annotations on more than one test cases (same cases as presented in HIT description); not following instructions."
    row_indexer = batch["WorkerId"].isin(really_bad_ones) == True
    batch.loc[row_indexer, "Reject"] = rejection_text
    
    # accept others
    row_indexer = batch["WorkerId"].isin(really_bad_ones) == False
    batch.loc[row_indexer, "Approve"] = "X"
    
    if args.output:
        batch.to_csv(f"{args.batch_results}"+"_rejections.csv", index=False)

    print("--- Without bad workers ---")
    clean = df[~df["WorkerId"].isin(really_bad_ones)]
    print(df.shape, clean.shape)
    print(clean.annotation.value_counts())
    grp = clean.groupby(["facet_desc", "question", "answer_1", "answer_2"]).agg(list)
    gpt3, gpt2, tie = 0, 0, 0
    off = 0
    for k, v in grp["annotation"].value_counts().items():
        if len(k) in [2, 4]:
            cs = Counter(k)
            if cs.get("gpt3", 0) > cs.get("gpt2", 0):
                gpt3 += v
            elif cs.get("gpt3", 0) < cs.get("gpt2", 0):
                gpt2 += v
            else:
                tie += v
        elif len(k) in [3, 5]:
            cs = Counter(k[:-1]) # kick out last annotation
            if cs.get("gpt3", 0) > cs.get("gpt2", 0):
                gpt3 += v
            elif cs.get("gpt3", 0) < cs.get("gpt2", 0):
                gpt2 += v
            else:
                tie += v
        else:
            off += v
    #print(grp["annotation"].value_counts())
    print(f"GPT3: {gpt3} | GPT2: {gpt2} | Tie: {tie} | off: {off}")

if __name__=="__main__":

    parser = argparse.ArgumentParser("MTurk")
    #parser.add_argument("--input_csv", help="Input csv for mturk study.")
    parser.add_argument("--batch_results", help="Annotated batch from MTurk.")
    parser.add_argument("--output", help="Write rejections or not.", default="")
    args = parser.parse_args()
    main(args)