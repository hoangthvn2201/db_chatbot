from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
import torch

# 1. Model and Tokenizer Setup
model_path = 'phamhai/Llama-3.2-1B-Instruct-Frog'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)

# 2. Pipeline Configuration
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    do_sample=True,
    temperature=0.7,
    top_p=0.95,
    device='cuda:0'
)

# 3. Chatbot Class
class ContextAwareChatbot:
    def __init__(self, pipeline,max_history: int = 5):
        self.pipeline = pipeline
        self.model = AutoModelForCausalLM.from_pretrained('phamhai/Llama-3.2-1B-Instruct-Frog')
        self.tokenizer = AutoTokenizer.from_pretrained('phamhai/Llama-3.2-1B-Instruct-Frog')
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        
    def _build_prompt(self) -> str:
        # Build context from history
        history_text = ""
        for exchange in self.conversation_history[-self.max_history:]:
            history_text += f"Human: {exchange['human']}\nAssistant: {exchange['assistant']}\n\n"
        
        # Create the full prompt with context
        prompt = f"""Bạn là một trợ lí ảo thông minh có thể trả lời những câu hỏi của người dùng. Dựa vào đoạn hội thoại trong quá khứ, cố gắng trả lời câu hỏi người dùng một cách chính xác và trung thực nhất.
Đoạn chat trong quá khứ:     
<START_OF_HISTORY_CONTEXT>
{history_text}
<END_OF_HISTORY_CONTEXT>"""
        return prompt
    
    def _clean_response(self, response: str) -> str:
        # Clean up the generated response
        response = response.split("Assistant:")[-1].strip()
        # Stop at any new "Human:" or "Assistant:" markers
        if "Human:" in response:
            response = response.split("Human:")[0].strip()
        return response
    
    def chat(self, user_input: str) -> str:
        # Generate the contextualized prompt
        prompt = self._build_prompt()
        
#         # Generate response
#         response = self.pipeline(
#             prompt,
#             return_full_text=False,
#             clean_up_tokenization_spaces=True
#         )[0]['generated_text']
        
#         # Clean the response
#         cleaned_response = self._clean_response(response)
        messages =[
            {'role':'system',
             'content':prompt}
            ,
            {'role':'user',
             'content':user_input}
        ] 
        #role: system, user, assistant
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors='pt')
        outputs = self.model.generate(tokenized_chat, max_new_tokens=256).to('cuda')
        bot_response = self.tokenizer.decode(outputs[0])
        bot_response = bot_response.split('<|start_header_id|>assistant<|end_header_id|>')
        bot_response = bot_response[1].strip()[:-10]
        # Update conversation history
        self.conversation_history.append({
            'human': user_input,
            'assistant': bot_response
        })
        
        return bot_response
    
    def get_history(self) -> List[Dict[str, str]]:
        return self.conversation_history
    
    def clear_history(self):
        self.conversation_history = []

# 4. Create chatbot instance
chatbot = ContextAwareChatbot(pipe)

# 5. Example usage function
def chat_session():
    print("Chatbot initialized. Type 'exit' to end the conversation, 'clear' to clear history.")
    
    while True:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        elif user_input.lower() == 'clear':
            chatbot.clear_history()
            print("Conversation history cleared!")
            continue
            
        response = chatbot.chat(user_input)
        print(f"\nAssistant: {response}")

# 6. Example of how to use
if __name__ == "__main__":
    chat_session()