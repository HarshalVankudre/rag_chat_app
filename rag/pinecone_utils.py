from typing import Dict, Iterable, List, Optional
from openai import OpenAI
from pinecone import Pinecone

def get_openai_client(api_key: str, base_url: Optional[str] = None) -> OpenAI:
    kwargs = {"api_key": api_key}
    if base_url: kwargs["base_url"] = base_url.strip()
    return OpenAI(**kwargs)

def get_pinecone_index(api_key: str, host: Optional[str], index_name: Optional[str]):
    pc = Pinecone(api_key=api_key)
    if host and host.strip():
        return pc.Index(host=host.strip())
    if index_name and index_name.strip():
        return pc.Index(index_name.strip())
    raise ValueError("Provide Pinecone index host or index name.")

def embed_texts(client: OpenAI, texts: List[str], model: str) -> List[List[float]]:
    resp = client.embeddings.create(model=model, input=texts)
    return [list(map(float, d.embedding)) for d in resp.data]

def retrieve_chunks(index, query_vec: List[float], top_k: int, namespace: Optional[str]) -> Dict:
    return index.query(vector=[float(x) for x in query_vec], top_k=int(top_k), include_values=False, include_metadata=True, namespace=(namespace or "__default__"))

def stream_chat_completion(client: OpenAI, model: str, messages: List[Dict], temperature: float) -> Iterable[str]:
    stream = client.chat.completions.create(model=model, messages=messages, temperature=temperature, stream=True)
    for event in stream:
        delta = event.choices[0].delta.content or ""
        if delta: yield delta
