# packages
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp
from pprint import pprint 

# download the model from HuggingFace
# https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true
# save in subfolder "models"

# callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# llm = LlamaCpp(
# 	model_path = "./models/llama-2-7b-chat.Q3_K_M.gguf",
# 	temperature=0.75,
# 	max_tokens=2000,
# 	top_p=1,
# 	callback_manager=callback_manager,
# 	verbose=True, # Verbose is required to pass to the callback manager
# 	# device="cpu"
# )

# # system and user prompt
# system_prompt = "Eve lives in Hamburg.; Bob lives in Cape Town.; Alice lives in Mumbay."
# user_prompt = "Where does Eve live?"

# # naive approach
# prompt_naive = f"{system_prompt}\n{user_prompt}"

# llm(prompt_naive)

# # set up prompt correctly
# llama_prompt = f"<s>[INST]<<SYS>>\n{system_prompt}<</SYS>>\n{user_prompt}[/INST]"
# pprint(llama_prompt)

# # run Llama2
# res = llm(llama_prompt)
# print(res)

from llama_cpp import Llama

# Put the location of to the GGUF model that you've download from HuggingFace here
model_path = "./models/llama-2-7b-chat.Q3_K_M.gguf"

# Create a llama model
model = Llama(model_path=model_path)

# Prompt creation
system_message = "You are a helpful assistant"
user_message = "Generate a list of 5 funny dog names"

prompt = f"""<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]"""

# Model parameters
max_tokens = 100

# Run the model
output = model(prompt, max_tokens=max_tokens, echo=True)

# Print the model output
print(output)