from google import genai

client = genai.Client(api_key="AIzaSyArWmLsYKgBvGrK5_pQ3oCUu6FD1Pqo14g")



response = client.chat.completions.create(
    model="o3-mini-2025-01-31",  # Ensure the correct model name
    messages=[{"role": "user", "content": "Explain how AI works"}]
)

raw_output = response.choices[0].message.content

print(response.text)