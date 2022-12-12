import os
import openai
from tenacity import retry, wait_exponential
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

@retry(wait=wait_exponential(multiplier=1, min=4, max=10))
def ping_GPT3(prompt: str):
    """Ping GPT3 with a prompt. Retries until a response is received"""
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=50,
        top_p=1,
        frequency_penalty=0.2,
        presence_penalty=0.5,
        temperature=0.5,
    )

    response = response.choices[0].text.strip()
    return response.split("\n")[0]
