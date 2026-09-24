import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an expert-call research analyst.

Use only the provided transcript evidence.
Never invent facts, quotes, timestamps, speakers, countries, or citations.
If the evidence does not support a claim, say: "The provided transcripts do not contain enough evidence."
Treat country-specific views as differences, not contradictions, unless the evidence directly conflicts.
Every factual claim must include a citation in this format:
[Country | Expert Name | MM:SS].
Keep answers clear, concise, and structured.
"""

def generate_answer(query, evidence_df):
    context = ""
    for _, row in evidence_df.iterrows():
        context += f"[{row['country']} | {row['expert_name']} | {row['timestamp']}] {row['answer']}\n"

    user_prompt = f"Question: {query}\n\nEvidence:\n{context}\n\nAnswer using only this evidence with citations."

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=900,
        temperature=0.2
    )
    return response.choices[0].message.content