# ConvSim

Code for the SIGIR 2023 paper "Exploiting Simulated User Feedback for Conversational Search: Ranking, Rewriting, and Beyond".

## Overview

![the_image-3](https://user-images.githubusercontent.com/28223751/233872576-deb20ee7-e05e-4031-9702-367c4400d118.png)

We develop a comprehensive experimental framework based on simulated user–system interactions that allow us to evaluate multiple state-of-the-art mixed-initiative CS systems, addressing several challenges, such as contextual query resolution, asking clarifying questions, and incorporating user feedback.

At the center of this framework is ConvSim, an LLM based user simulator capable of answering clarification questions and giving explicit feedback.
Leveraging the outputs of ConvSim over multiple rounds of feedback, we improve the retrieval effectiveness of several state-of-the-art conversational search systems on the [CAsT benchmark](https://github.com/daltonj/treccastweb) and generate [a dataset of 30k conversations](https://drive.google.com/drive/folders/1wB2cyU3k-v00rO0du_j9-QgTNR_cSJGf?usp=sharing) useful for tasks such as discourse-aware query rewriting, re-ranking, and much more.

## How to Run

Clone the repository and create/activate a virtual environment.

```
python3 -m venv convsim
source convsim/bin/activate
```

Install all necessary libraries

```
pip install -r requirements.txt
```

Download CAsT benchmark and other necessary artifacts like a subset of the indexed document collection.

```
bash setup.sh
```

Create a `.env` file with your openai API key. It should look something like:
```
# API Key for GPT3
OPENAI_API_KEY="api_key"
```

Run.

```
python main.py
```

## Future Work

We intend on turning this framework into a python package that anyone can experiment with.
This package will help us demo the utility of this framework to non-conversational IR datasets.
