import requests
import json
from typing import Callable, Generator

def ask_llm(base_url: str, api_key: str, model: str, query: str, retrieved_docs: str, is_stream: bool = True) -> str | Callable[[], Generator[str, None, None]]:

    prompt = f"""
    You are an expert Q&A system that is trusted around the world for your factual accuracy.
    Always answer the query using the provided context information, and not prior knowledge. Ensure your answers are fact-based and accurately reflect the context provided.
    Some rules to follow:
    1. Never directly reference the given context in your answer.
    2. Avoid statements like 'Based on the context, ...' or 'The context information ...' or anything along those lines.
    3. Focus on succinct answers that provide only the facts necessary, do not be verbose. Your answers should be max five sentences, up to 300 characters.
    4. Use the same language as the query.
    ---------------------
    Context: {retrieved_docs}
    ---------------------
    Given the context information and not prior knowledge, answer the query.
    Query: {query}
    """

    header = {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + api_key
    }

    context = [{"role": "user", "content": prompt}]
    data = {"model": model, "messages": context, "stream": is_stream}
    url = base_url + "/completions"

    try:
        response = requests.request(
            "POST",
            url,
            headers=header,
            json=data,
            stream=is_stream,
        )

        if not is_stream:
            if not response.ok:
                print(f"Error during requesting: {response.status_code}")
                return ''
            result = response.json()
            return result['choices'][0]['message']['content'].strip()
        else:
            def generate():
                i = 0
                for line in response.iter_lines():
                    line_str = str(line, encoding="utf-8")
                    if line_str.startswith("data:") and line_str[5:]:
                        if line_str.startswith("data: [DONE]"):
                            break
                        line_json = json.loads(line_str[5:])
                        if "choices" in line_json:
                            if len(line_json["choices"]) > 0:
                                choice = line_json["choices"][0]
                                if "delta" in choice:
                                    delta = choice["delta"]
                                    if "content" in delta:
                                        delta_content = delta["content"]
                                        i += 1
                                        if i < 40:
                                            print(delta_content, end="")
                                        elif i == 40:
                                            print("......")
                                        yield delta_content

                    elif len(line_str.strip()) > 0:
                        print(line_str)
                        yield line_str
            return generate

    except Exception as e:
        print(f"Error during requesting: {e}")
        return ''
