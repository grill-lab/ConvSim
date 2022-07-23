from simulated_users import SimulatedUser
from utils import SearchSystem
from pathlib import Path

if __name__ == "__main__":

    topic = "Watergate Scandal"
    trained_on = "cast"

    simuated_user = SimulatedUser(
        topic=topic,
        facets=[],
        prior_knowledge=[],
        patience=10,
        model_path="/shared/models/USi-finetuned-cast/simplet5-epoch-19-train-loss-0.2951-val-loss-4.134"
    )
    search_system = SearchSystem("/index")
    session = {
        'closed' : False,
        'conversation_history' : []
    }

    while simuated_user.patience > 0 and not session['closed']:
        user_utterance = simuated_user.generate_utterance(session['conversation_history'])
        system_response = ''

        if 'bye' in user_utterance.lower() or 'thank' in user_utterance.lower():
            session['closed'] = True
            system_response = "Okay. Bye!"
        
        else:
            rewrite_context = []
            for turn in session['conversation_history']:
                rewrite_context.extend([turn['user_utterance'], turn['system_response']])
            rewrite = search_system.rewrite_query(user_utterance, rewrite_context)
            candidate_documents = search_system.retrieve_documents(rewrite)
            ranked_passages = search_system.rank_passages(rewrite, candidate_documents)
            # Generate Summary
            system_response = search_system.generate_summary(
                ranked_passages, 2, truncation=True
            )
        
        print(f"User: {user_utterance}")
        print(f"System: {system_response}")
        session['conversation_history'].append({
            "user_utterance": user_utterance,
            "system_response": system_response
        })
    
    conversations_dir = f"/shared/conversations/{trained_on}"
    Path(conversations_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{conversations_dir}/{topic}.txt", "w") as outfile:
        for idx, turn in enumerate(session['conversation_history']):
            outfile.write(f"Turn: {idx+1}\n")
            outfile.write(f"User: {turn['user_utterance']}\n")
            outfile.write(f"System: {turn['system_response']}\n")
            outfile.write("\n")

