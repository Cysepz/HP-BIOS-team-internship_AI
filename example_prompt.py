import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def print_blue(text):
    print(f"\033[94m{text}\033[0m")
    
# Define local model & tokenizer path
model_path = "C:/Users/peggyzhu/Github repo/codellama/CodeLlama-7b"
model_name = "meta-llama/CodeLlama-7b-hf"

print_blue("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")

code_snippet = """
def exe_query(user_input):
    query = "SELECT * FROM users WHERE username = '" + user_input + "';"
    execute(query)
"""

prompt = f"""
Please judge if the following code snippet is vulnerable, and provide a brief explanation if necessary.
{code_snippet}

Response format:
- Yes/No
- Vulnerability type (if applicable)
- Explanation (if necessary)
"""

inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print_blue("Generating response...")
outputs = model.generate(
    inputs["input_ids"],
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.9,
    repetition_penalty=1.2,
)

response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print_blue("Model response:")
print_blue(response)