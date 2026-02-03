# test_model.py
from app import llm

# Test the model directly
response = llm("hello how are you what is your name")
print("Model response:", response)