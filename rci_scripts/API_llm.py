from openai import OpenAI

client = OpenAI(base_url="http://<NODE>:8333/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "system", "content": "You are a fact-checking assistant."},
        {"role": "user", "content": "The Czech Republic has the highest beer consumption per capita in the world."},
    ],
)
print(response.choices[0].message.content)

"""You can find your compute node name by checking:

hostname

inside the running job or in your SLURM log (e.g. n21), then replace <NODE> with that (like http://n21:8333/v1)."""