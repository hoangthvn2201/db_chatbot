from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# 1. Model and Tokenizer Setup
model_path = 'phamhai/Llama-3.2-1B-Instruct-Frog'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
