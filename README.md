# Overview

USi 2.0 is a user simulator that anyone (researchers, students, industry practitioners..) can use to evaluate the effectiveness of their search systems or generate annotated synthetic conversations for further analysis.

USi is configurable. End users can determine the “type” of user they want to simulate in regards to patience levels, personality (for user revealment), and topic knowledge.

To use USi 2.0, one needs a search system, an index, a topic of interest, a list of sub-topics (facets), and a qrels file (with relevance judgements for each facet). Systems should be able to take in an utterance (question, feedback, comment…) from USi and return a response (question, suggestion, answer) and an optional (depending on response type) ranked list of documents (provenance). Search systems should ideally have a summarization component.

At the end of each conversation, USi scores the conversation using conversation-level metrics and taking the user state into account(i.e number of sub-topics explored, user’s current patience level / 1.0 etc).

USi should provide rapid responses and not require heavy compute to use.

## Data Generation

Document collection index is built outside of the system and attached as a volume.

To run, do:

`INDEX_PATH=path/to/index docker-compose up --build`
