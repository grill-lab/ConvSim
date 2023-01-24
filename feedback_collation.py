import shelve
from src.data_classes.conversational_turn import Document, ConversationalTurn
from transformers import pipeline
import json
import math

sentiment_analysis = pipeline("sentiment-analysis", device=0)
feedback_at_each_turn = dict()

POSITIVE = 'POSITIVE'
NEGATIVE = 'NEGATIVE'

with shelve.open('data/generated_files/gpt3-simulator/runs') as runs:
    for run_name in runs:
        try:
            conversational_turn_objects = runs[run_name]
        except:
            continue
        # print(run_name)
        for ct_obj in conversational_turn_objects:
            precision_at_1 = ct_obj.evaluate_turn(measure='P(rel=2)@1')
            precision_at_3 = ct_obj.evaluate_turn(measure='P(rel=2)@3')
            
            feedbacks = [turn['utterance'] for turn in ct_obj.conversation_history if turn['utterance_type'] == 'feedback']
            if ct_obj.user_utterance_type == 'feedback':
                feedbacks += [ct_obj.user_utterance]
            
            if len(feedbacks) == 0:
                continue

            if ct_obj.turn_id not in feedback_at_each_turn:
                feedback_at_each_turn[ct_obj.turn_id] = {POSITIVE: [], NEGATIVE: []}
            
            sentiment_results = sentiment_analysis(feedbacks)
            # print(ct_obj.turn_id)
            # print(ct_obj.system_response)
            for idx, f_res in enumerate(zip(feedbacks, sentiment_results)):
                feedback, res = f_res
                if (precision_at_3 > 0.3) and res['label'] == POSITIVE:
                    feedback_at_each_turn[ct_obj.turn_id][POSITIVE].append(feedback)
                
                # elif (precision_at_3 > 0.3) and res['label'] == POSITIVE:
                #     continue

                elif (precision_at_3 == 0.0) and res['label'] == NEGATIVE:
                    feedback_at_each_turn[ct_obj.turn_id][NEGATIVE].append(feedback)
                
                elif (math.isnan(precision_at_3)):
                    feedback_at_each_turn[ct_obj.turn_id][res['label']].append(feedback)
                # else:
                #      feedback_at_each_turn[ct_obj.turn_id][NEGATIVE].append(feedback)

with open('feedback_utterances.json', 'w') as feedback_file:
    json.dump(feedback_at_each_turn, feedback_file, indent=4, ensure_ascii=False)