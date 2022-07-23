import json
import os
from typing import List
from pathlib import Path
from utils import SearchSystem

if __name__ == "__main__":

    index_dir: str = '/index'
    topics_dir: str = '/shared/topics'
    data_output_dir: str = '/shared/training_data'

    Path(data_output_dir).mkdir(parents=True, exist_ok=True)
    
    topic_count = 1
    passage_threshold = 3
    all_topics: List = []
    search_system = SearchSystem(index_dir)

    for filename in os.listdir(topics_dir):
        topics_set_file_path = os.path.join(topics_dir, filename)
        with open(topics_set_file_path) as topics_set_file:
            topics_set = json.load(topics_set_file)
            for topic in topics_set:
                revised_topic = []
                for turn_index, turn in enumerate(topic['turn']):
                    # Extract Turn ID
                    turn_id = f"{topic['number']}_{turn['number']}"
                    print(f"Turn ID: {turn_id}")
                    # Extract Utterance
                    raw_utterance = turn['raw_utterance']
                    print(f"Utterance: {raw_utterance}")
                    # Extract Context
                    dialogue_history = revised_topic[:turn_index]
                    context = []
                    for history in dialogue_history:
                        context.extend([history['utterance'], history['system_response']])
                    # Rewrite Raw Utterance
                    rewrite = search_system.rewrite_query(raw_utterance, context)
                    # Retrieve documents
                    candidate_documents = search_system.retrieve_documents(rewrite)
                    # Extract and rank passages
                    ranked_passages = search_system.rank_passages(rewrite, candidate_documents)
                    # Generate Summary
                    system_response = search_system.generate_summary(
                        ranked_passages, passage_threshold, truncation=True
                    )
                    print(f"System Response: {system_response}")
                    # Add topic to revised_topic
                    revised_topic.append({
                        "turn_id": turn_index,
                        "utterance": raw_utterance,
                        "rewrite": rewrite,
                        "system_response": system_response,
                        "provenance": [passage['id'] for passage in ranked_passages[:3]]
                    })
            
                all_topics.append({
                    "topic_id": topic_count,
                    "turns": revised_topic 
                })

                # update topic count
                topic_count += 1
    
    with open(f"{data_output_dir}/conversations.json", "w") as output_file:
        json.dump(all_topics, output_file, indent=4)
