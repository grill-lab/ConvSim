# %%
import json
import tqdm
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, wait_exponential


# %% [markdown]
# #### Collect Clarifying Questions and Rewrites

# %%
dataset = []
# Load the base data.
with open('../../data/datasets/cast/year_4/annotated_topics.json', 'r') as f:
    annotated_topics = json.load(f)

rewrite_system_prompt = "The following is a conversation between a user and a search system to learn more about '$information_need'. Your job is to rewrite the user's last utterance to the system so that it is free of ambiguity and the search system is able to answer it better."
cq_system_prompt = "The following is a conversation between a user and a search system to learn more about '$information_need'. Based on the user's last utterance, your job is to write a clarification question that the search system should ask the user in order to clarify their information need and retrieve better quality results."
summary_prompt = "The following is a conversation between a user and a search system to learn more about '$information_need'. Your task is to write a summary of the conversation that describes what the user and system have said to each other. Do not generate an answer to the User's question."

# %%
load_dotenv()
client = OpenAI()

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def get_message(conversation_str: str, system_prompt: str):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": conversation_str
            }
        ],
        temperature=0.1,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    return response.choices[0].message.content

# %%
from string import Template

stash = {}
parsed_turns = set()
for topic in tqdm.tqdm(annotated_topics):
    for index, turn in enumerate(topic['turn']):
        turn_id = f"{topic['number']}_{turn['number']}"
        if turn_id in parsed_turns:
            continue
        information_need = turn.get("information_need")
        conversation_str = ""
        for previous_turn in topic['turn'][:index]:
            conversation_str += f"User: {previous_turn.get('utterance')}\n"
            conversation_str += f"System: {previous_turn.get('response')}\n"
        
        # Get summary.
        prompt = Template(summary_prompt).substitute(
            information_need=information_need)
        summary = get_message(conversation_str, prompt)
        conversation_str += f"User: {turn.get('utterance')}\n"

        # Get rewrite.
        prompt = Template(rewrite_system_prompt).substitute(
            information_need=information_need)
        rewrite = get_message(conversation_str, prompt)

        # Get clarification question.
        prompt = Template(cq_system_prompt).substitute(
            information_need=information_need)
        cq = get_message(conversation_str, prompt)

        # Store results in stash
        stash[turn_id] = {
            'rewrite': rewrite,
            'cq': cq,
            'summary': summary
        }
        

# %%
with open('../../data/datasets/cast/year_4/gold_questions_and_rewrites.json', 
          'w') as f:
    json.dump(stash, f, indent=4)


