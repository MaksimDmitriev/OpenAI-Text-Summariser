import time

import openai
import tiktoken
from project_utils import read_file

from openai import OpenAI


def summarize_document(document, chunk_size=1000, overlap=200):
    """Summarizes a given document using the OpenAI API.

    Args:
      document: The text to be summarized.
      chunk_size: The maximum size of each document chunk.
      overlap: The number of overlapping tokens between chunks.

    Returns:
      The summarized text.
    """

    def get_encoding():
        return tiktoken.encoding_for_model("gpt-3.5-turbo")

    def count_tokens(text):
        encoding = get_encoding()
        return len(encoding.encode(text))

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

    def summarize_chunk(chunk):
        try:
            client = OpenAI()
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # Replace with your desired model
                # prompt=f"Summarize this text: {chunk}",
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
            return summarize_chunk(chunk)
        except Exception as e:
            print(f"Error summarizing chunk: {e}")
            return ""

    chunks = split_document(document, chunk_size, overlap)
    summaries = [summarize_chunk(chunk) for chunk in chunks]
    return "\n".join(summaries)


content = read_file("long_text.txt")
print(summarize_document(document=content))
