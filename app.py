import os
from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="API GPT")
class Query(BaseModel):
    question: str

@app.post("/ask")
def ask_model(query: Query):
    """
    Endpoint que recebe uma pergunta em texto e responde usando agente.
    """
    try:
        response = client.responses.create(
            model="gpt-4o-mini",
            input=query.question,
        )


        answer = getattr(response, "output_text", None)
        if not answer:
            # fallback - dict-like
            if isinstance(response, dict):
                answer = response.get("output", "")
            else:
                answer = str(response)

        return {"answer": answer}
    except Exception as e:
        return {"error": str(e)}
