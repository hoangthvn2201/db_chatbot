from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer


translation_model_name = 'VietAI/envit5-translation'
translation_tokenizer = AutoTokenizer.from_pretrained(translation_model_name)
translation_model = AutoModelForSeq2SeqLM.from_pretrained(translation_model_name).to('cuda')

sql_coder_name = "defog/sqlcoder-7b-2"
sql_tokenizer = AutoTokenizer.from_pretrained(sql_coder_name)
sql_model = AutoModelForCausalLM.from_pretrained(
    sql_coder_name,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    device_map='auto'
)
embedding_model_name = 'mixedbread-ai/mxbai-embed-large-v1'
embedding_model = SentenceTransformer(embedding_model_name, device='cuda')
