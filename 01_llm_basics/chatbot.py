import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

print("AI Chatbot started!")
print("Type 'exit' to quit.\n")

messages = [
    {
        "role": "system",
        "content": "You are a helpful Computer Networks tutor. Explain concepts clearly using simple examples."
    }
]

while True:
    user_input = input("You: ")

    if user_input.lower() == "exit":
        print("Goodbye!")
        break

    messages.append({
        "role": "user",
        "content": user_input
    })

    response = client.chat.completions.create(
        model="google/gemini-3.7-flash",
        max_tokens=2000,
        messages=messages,
        stream=True
    )

    assistant_message = ""
    print("AI: ", end="")
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            print(content, end="")
            assistant_message += content

    messages.append({
            "role": "assistant",
            "content": assistant_message
        })
    print()