import time
import openai
import tiktoken
from project_utils import read_file
from openai import OpenAI

model_name = "gpt-3.5-turbo"
client = OpenAI()


def get_encoding():
    return tiktoken.encoding_for_model(model_name)


def split_document(document, chunk_size, overlap):
    encoding = get_encoding()
    tokens = encoding.encode(document)
    chunks = []
    i = 0
    while i < len(tokens):
        j = min(i + chunk_size, len(tokens))
        chunk = encoding.decode(tokens[i:j])
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def summarize_chunk(chunk, open_ai_client):
    try:
        response = open_ai_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": f"Please summarize the following text:\n\n{chunk}"}
            ],
            max_tokens=100,
            n=1,
            stop=None,
            temperature=0.5
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        # Handle rate limit errors, e.g., exponential backoff
        time.sleep(60)
        return summarize_chunk(chunk, open_ai_client)
    except Exception as e:
        print(f"Error summarizing chunk: {e}")
        return ""


def summarize_document(document, open_ai_client, chunk_size=1000, overlap=200):
    chunks = split_document(document, chunk_size, overlap)
    summaries = [summarize_chunk(chunk, open_ai_client) for chunk in chunks]
    return "\n".join(summaries)


content = read_file("long_text.txt")
print(summarize_document(document=content, open_ai_client=client))
