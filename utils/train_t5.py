# %% [markdown]
# ## Runs Analysis

# %%
import sys
sys.path.append("/home/ubuntu/ConvSim")

# %%
from src.data_classes import ConversationalTurn
import shelve

# %%
dataset = []

# %%
run_name = "cast_cq_with_feedback"
with shelve.open(f"../data/generated_conversations/{run_name}/turns_db") as db:
    for turn_id in db:
        conversational_turn = db[turn_id]
        if conversational_turn.user_utterance_type == "feedback":
            # build conversation list
            conversation = []
            for historical_turn in conversational_turn.conversation_history:
                conversation.append(historical_turn['utterance'])
            conversation.append(conversational_turn.user_utterance)
            rewritten_utterance = conversational_turn.rewritten_utterance.replace("USER: ", "").strip()
            dataset.append({
                "conversation": conversation,
                "rewrite": rewritten_utterance,
            })

# %%
run_name = "cast_rewrites_no_feedback"
with shelve.open(f"../data/generated_conversations/{run_name}/turns_db") as db:
    for turn_id in db:
        conversational_turn = db[turn_id]
        conversation = []
        for historical_turn in conversational_turn.conversation_history:
            conversation.append(historical_turn['utterance'])
        conversation.append(conversational_turn.user_utterance)
        rewritten_utterance = conversational_turn.rewritten_utterance.replace("USER: ", "").strip()
        dataset.append({
            "conversation": conversation,
            "rewrite": rewritten_utterance,
        })

# %%
run_name = "cast_response_with_feedback"
with shelve.open(f"../data/generated_conversations/{run_name}/turns_db") as db:
    for turn_id in db:
        conversational_turn = db[turn_id]
        conversation = []
        for historical_turn in conversational_turn.conversation_history:
            conversation.append(historical_turn['utterance'])
        conversation.append(conversational_turn.user_utterance)
        rewritten_utterance = conversational_turn.rewritten_utterance.replace("USER: ", "").strip()
        dataset.append({
            "conversation": conversation,
            "rewrite": rewritten_utterance,
        })

# %%
print(dataset[0])
print(len(dataset))

# %%
from transformers import AutoTokenizer
from datasets import Dataset

MODEL_NAME = "t5-base" #"t5-base" 

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, truncation_size="left")
dataset = Dataset.from_list(dataset)
# dataset = dataset.select(range(10000))
dataset = dataset.shuffle(seed=42)
dataset = dataset.train_test_split(test_size=0.1)

# %%
from transformers import AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer
import evaluate
import numpy as np

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=MODEL_NAME)
rouge = evaluate.load("rouge")

# %%
# prefix = "The following is a conversation between a user and a search system. Your job is to rewrite the user's last utterance so that it is free of ambiguity and the search system is able to answer it better.\n\n"
prefix = "reformulate: "
# def create_conversation(conversation):
#     parsed_conversation = ""
#     for idx, turn in enumerate(conversation):
#         if idx % 2 == 0:
#             parsed_conversation += f"USER: {turn}\n"
#         else:
#             parsed_conversation += f"SYSTEM: {turn}\n"
#     return parsed_conversation


def preprocess_function(examples):
    parsed_conversations = [prefix + " </s> ".join(conversation) for conversation in examples['conversation']]
    # parsed_conversations = [prefix + create_conversation(conversation) for conversation in examples['conversation']]
    print(parsed_conversations[0])
    print(examples['rewrite'][0])
    inputs = [prefix + conv for conv in parsed_conversations]
    model_inputs = tokenizer(inputs, max_length=1024, truncation=True)

    labels = tokenizer(text_target=examples["rewrite"], max_length=32, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = rouge.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}


# %%
tokenized_dataset = dataset.map(preprocess_function, batched=True)

# %%
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, device_map="auto")

# %%
MODEL_NAME = MODEL_NAME.replace("/", "-")
TUNED_MODEL_NAME = f"../../data/models/tuned-{MODEL_NAME}-rewriter-v2-2e3-20epochs"

training_args = Seq2SeqTrainingArguments(
    output_dir=TUNED_MODEL_NAME,
    evaluation_strategy="steps",
    eval_steps=20,
    learning_rate=2e-3, #2e-52
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=4, # 4
    weight_decay=0.01,
    save_total_limit=1,
    num_train_epochs=20,
    save_steps=20,
    save_strategy="steps",
    predict_with_generate=True,
    generation_max_length=32,
    #generation_num_beams=4,
    fp16=True,
    # auto_find_batch_size=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_rougeL",
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# %%
trainer.train()

# %%
from transformers import pipeline

rewriter = pipeline("text2text-generation", model=trainer.model, tokenizer=trainer.tokenizer)

# %%
conversation = ["Can you help me find a diet for myself?", "What kind of diet do you want?", "I want something that is vegan-friendly, maintainable and not very hard to keep up"]
# conversation = create_conversation(conversation)
conversation = " </s> ".join(conversation)
conversation = prefix + conversation
x = rewriter(conversation, max_length=64)
print(x[0]['generated_text'])

# %%
trainer.save_model()
