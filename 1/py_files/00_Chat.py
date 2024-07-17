from openai import OpenAI
import os 

client = OpenAI(
	api_key=os.environ.get("OPENAI_API_KEY"),
)

completion = client.chat.completions.create(
	messages=[
		{
			"role":"user",
			"content":"The coffee is extremely ...",
		}
	],
	model="gpt-3.5-turbo"
)

print(completion.choices[0].message.content)