from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import warnings
# model_path = 'phamhai/Llama-3.2-1B-Instruct-Frog'
device = torch.device('cuda')
model_path = 'huyhoangt2201/llama-3.2-1b-sql_finetuned_billingual_3.0_merged'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')

prompt_template = """
You are an SQL query assistant. Based on the table information below, generate an SQL query to retrieve the relevant information for the user. If the user’s question is unrelated to the table, respond naturally in user's language.

The jidouka table contains the following columns:

id: Row identifier (int)
tên_cải_tiến: Name of the improvement (str)
loại_hình_công_việc: Type of work that the improvement is intended to enhance (str) (e.g., database processing, data entry, workflow optimization, etc.)
công_cụ: Tool used to achieve the improvement (str) (e.g., Python, Excel, Visual Studio Code, etc.)
mô_tả: Detailed description of the improvement (str) (e.g., each step of the improvement process)
sản_phẩm: Output product of the improvement (str) (e.g., .csv file, .xlsx file, etc.)
tác_giả: Contributor, company employee, or creator of the improvement (str)
bộ_phận: Department of the author, usually referred to as "dc" (str) (e.g., dc1, dc2, dc3, dcd, souko, etc.)
số_giờ: Number of hours saved by applying the improvement (int)
số_công_việc_áp_dụng: Number of tasks in the company that the improvement has supported (int)
thời_điểm_ra_mắt: Launch date of the tool (str) (e.g., 2024-10-11, 2024-10-09, etc.)
thông_tin_thêm: Link to additional documentation (PowerPoint, video) on using the improvement or the improvement’s tool (str)
"""

from typing import List, Dict
class ContextAwareChatbot:
    def __init__(self,prompt, max_history: int = 5):
        self.model = model
        self.tokenizer = tokenizer
        self.max_history = max_history
        self.conversation_history: List[Dict[str, str]] = []
        self.prompt=prompt
    def _build_prompt(self) -> str:
        # Build context from history

        return self.prompt

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
    
        eot = "<|eot_id|>"
        eot_id = self.tokenizer.convert_tokens_to_ids(eot)
        self.tokenizer.pad_token = eot
        self.tokenizer.pad_token_id = eot_id
        tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors='pt')
        inputs = self.tokenizer(tokenized_chat, padding=True, truncation=True, return_tensors='pt').to(device)
        attention_mask = inputs['attention_mask']
        outputs = self.model.generate(inputs['input_ids'],attention_mask=attention_mask,pad_token_id=self.tokenizer.eos_token_id, temperature=0.5,max_new_tokens=256).to(device)
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
chatbot = ContextAwareChatbot(prompt_template)

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
    warnings.filterwarnings("ignore")
