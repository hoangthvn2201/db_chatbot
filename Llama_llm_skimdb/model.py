from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from typing import List, Dict
import torch
import pandas as pd 

# # 1. Model and Tokenizer Setup
# model_path = 'phamhai/Llama-3.2-3B-Instruct-Frog'
# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
df = pd.read_csv('result.csv')
df.rename(columns={'pic':'contributor'}, inplace=True)

def format_table_for_chatbot(df):
    new_df = df[['innovation_name','task_type','tool','software','product','contributor','dc']]
    new_df.rename(columns={'innovation_name':'Tên cải tiến', 'task_type':'Loại hình công việc','tool':'Công cụ','software':'Phần mềm','product':'Thành phẩm','contributor':'Tác giả','dc':'Design Center'}, inplace=True)
    return new_df
df = format_table_for_chatbot(df)

# 3. Chatbot Class
class ContextAwareChatbot:
    def __init__(self,df, max_history: int = 5):
        self.pipeline = pipeline
        self.model = AutoModelForCausalLM.from_pretrained('phamhai/Llama-3.2-3B-Instruct-Frog')
        self.tokenizer = AutoTokenizer.from_pretrained('phamhai/Llama-3.2-3B-Instruct-Frog')
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.df = df 
    def _build_prompt(self) -> str:
        # Build context from history
        history_text = ""
        for exchange in self.conversation_history[-self.max_history:]:
            history_text += f"Human: {exchange['human']}\nAssistant: {exchange['assistant']}\n\n"
        table = self.df 
        # Create the full prompt with context
        prompt = f"""Bạn là một trợ lí ảo thông minh có thể trả lời những câu hỏi của người dùng. Dựa vào thông tin có trong bảng dưới và đoạn hội thoại trong quá khứ, cố gắng trả lời câu hỏi người dùng một cách chính xác và trung thực nhất.
        Bảng:
        <START_OF_TABLE>
        {table}
        <END_OF_TABLE>
        Đoạn chat trong quá khứ:
        <START_OF_HISTORY_CONTEXT>
        {history_text}
        <END_OF_HISTORY_CONTEXT>
        """
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
chatbot = ContextAwareChatbot(df)

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