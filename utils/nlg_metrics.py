# pip install jury bert-score
from jury import Jury
from typing import List, Tuple

def get_nlg_metrics(
        predictions: List[List[str]],
        references: List[List[str]],
        metrics: List[str] = ["bleu", "meteor", "bertscore"],
        clean: bool = True
        ) -> dict:
    """Compute several NLG metrics.

    Args:
        predictions: List of generated strings.
        references: List of ground truth strings.
        metrics: List of metrics to compute from Jury.
        clean (optional): Clean Jury dict. Defaults to True.
    Returns:
        A dictionary with metric names as keys and values as values. 
    """

    scorer = Jury(metrics=metrics)
    scores = scorer(predictions=predictions, references=references)

    if clean:
        return {m: scores[m]["score"] 
                for m in scores.keys() if type(scores[m]) != int}
    return scores

if __name__=="__main__":
    predictions = [
        ["the cat is on the mat", "There is cat playing on the mat"], 
        ["Look!    a wonderful day."]
    ]
    references = [
        ["the cat is playing on the mat.", "The cat plays on the mat."], 
        ["Today is a wonderful day", "The weather outside is wonderful."]
    ]

    # It's possible to have multiple generated strings or reference strings.
    # Thus List[List[str]].
    # If you're dealing with one string only (probs the case)
    # Then just wrap it into sublists: [[pred] for pred in predictions]

    print(get_nlg_metrics(predictions, references))
    
