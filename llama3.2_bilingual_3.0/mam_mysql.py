import mysql.connector
from mysql.connector import pooling
import pandas as pd
from typing import Union, List, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from contextlib import contextmanager
import logging
import os
os.environ['HF_HOME'] = "C:/Users/dmvns00008/.cache/"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseConfig:
    """Database configuration class"""
    def __init__(self, host: str, user: str, password: str, database: str, pool_size: int = 5):
        self.dbconfig = {
            "host": host,
            "user": user,
            "password": password,
            "database": database
        }
        self.pool_size = pool_size

class DatabaseConnection:
    """Handle MySQL connection pooling"""
    def __init__(self, config: DatabaseConfig):
        self.connection_pool = mysql.connector.pooling.MySQLConnectionPool(
            pool_name="mypool",
            pool_size=config.pool_size,
            **config.dbconfig
        )
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = self.connection_pool.get_connection()
        try:
            yield connection
        finally:
            connection.close()

class SQLAgent:
    """Generate SQL queries from natural language questions using LLM"""
    def __init__(self, model_name: str = "phamhai/Llama-3.2-1B-Instruct-Frog"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.context = """
        You are an SQL query assistant. Based on the table information below, generate an SQL query to retrieve the relevant information for the user. If the user's question is unrelated to the table, respond naturally in user's language.

        The jidouka table contains the following columns:
        +id: Row identifier (int)
        +tên_cải_tiến: Name of the improvement (str) 
        +loại_hình_công_việc: Type of work that the improvement is intended to enhance (str)
        +công_cụ: Tool used to achieve the improvement (str)
        +mô_tả: Detailed description of the improvement (str)
        +sản_phẩm: Output product of the improvement (str)
        +tác_giả: Contributor or creator of the improvement (str)
        +bộ_phận: Department of the author (str)
        +số_giờ: Number of hours saved (int)
        +số_công_việc_áp_dụng: Number of tasks supported (int)
        +thời_điểm_ra_mắt: Launch date of the tool (str)
        +thông_tin_thêm: Link to additional documentation (str)

        Guidelines:
        1. Use appropriate WHERE clauses for filtering
        2. Include ORDER BY for sorting when relevant
        3. Use appropriate aggregation functions (COUNT, AVG, SUM) when needed
        4. Use LIMIT when returning large result sets
        5. Use proper date formatting for date comparisons
        6. Always return meaningful column aliases
        
        Return format:
        - For SQL queries: Return only the SQL query
        - For non-SQL responses: Return "NOT_SQL_QUERY: " followed by explanation
        """
        
    def generate_query(self, question: str) -> str:
        """Generate SQL query from natural language question"""
        try:
            messages = [
                {'role': 'system', 'content': self.context},
                {'role': 'user', 'content': question}
            ]
            
            eot = "<|eot_id|>"
            eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id
            
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                tokenized_chat, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True
            )
            
            response = self.tokenizer.decode(outputs[0])
            response = response.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()[:-10]
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating SQL query: {e}")
            return "NOT_SQL_QUERY: Xin lỗi, tôi đang gặp vấn đề trong việc xử lý câu hỏi của bạn."

class ExecuteQueryAgent:
    """Execute SQL queries and handle responses"""
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        
    def is_valid_sql_query(self, query: str) -> bool:
        """Check if the string is a valid SQL query"""
        if query.startswith("NOT_SQL_QUERY:"):
            return False
            
        sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "JOIN"]
        query_upper = query.upper()
        return any(keyword in query_upper for keyword in sql_keywords)
    
    def sanitize_query(self, query: str) -> str:
        """Basic SQL injection prevention"""
        # Remove comments
        query = re.sub(r'/\*.*?\*/', '', query)
        query = re.sub(r'--.*$', '', query)
        
        # Remove multiple semicolons
        query = query.replace(';', '')
        
        return query
    
    def execute_query(self, query: str) -> Union[List[Dict[str, Any]], str]:
        """Execute SQL query and return results"""
        if not self.is_valid_sql_query(query):
            if query.startswith("NOT_SQL_QUERY:"):
                return query[13:].strip()
            return []
            
        try:
            sanitized_query = self.sanitize_query(query)
            with self.db_connection.get_connection() as connection:
                df = pd.read_sql_query(sanitized_query, connection)
                return df.to_dict('records')
        except Exception as e:
            logger.error(f"Error executing query: {e}")
            return []

class LLMAgent:
    """Generate natural language responses from query results"""
    def __init__(self, model_name: str = "phamhai/Llama-3.2-1B-Instruct-Frog"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_response(self, question: str, data_points: Union[List[Dict[str, Any]], str]) -> str:
        """Generate natural language response based on query results"""
        try:
            if isinstance(data_points, str):
                return data_points
                
            if not data_points:
                return "Tôi không tìm thấy dữ liệu phù hợp với câu hỏi của bạn. Vui lòng thử lại với câu hỏi khác."
            
            system_prompt = """You are an assistant who answers questions based on given data points and given schema.
            The jidouka table contains the following columns:
            +id: Row identifier (int)
            +tên_cải_tiến: Name of the improvement (str) 
            +loại_hình_công_việc: Type of work that the improvement is intended to enhance (str)
            +công_cụ: Tool used to achieve the improvement (str)
            +mô_tả: Detailed description of the improvement (str)
            +sản_phẩm: Output product of the improvement (str)
            +tác_giả: Contributor or creator of the improvement (str)
            +bộ_phận: Department of the author (str)
            +số_giờ: Number of hours saved (int)
            +số_công_việc_áp_dụng: Number of tasks supported (int)
            +thời_điểm_ra_mắt: Launch date of the tool (str)
            +thông_tin_thêm: Link to additional documentation (str)

            Here is the synonyms of name of each columns in jidouka, that will help you to generate more natural and concise answers:
            +tên_cải_tiến: improvement, innovation 
            +loại_hình_công_việc: task type, type of task, job type 
            +công_cụ: tool, toolkit, device, gadget
            +mô_tả: description, describe line, detail 
            +sản_phẩm: product, output, output product 
            +tác_giả: contributor, creator, employee
            +bộ_phận: department, design center 
            +số_giờ: saved hours, number of saved hours 
            +số_công_việc_áp_dụng: number of work, number of job 
            +thời_điểm_ra_mắt: launch time, creation time, time of release

            Requirements:
            - Always answer based on both human question, given datapoints.
            - Answer in the same language as the user's question
            - Be concise but informative
            - If the data points are empty, answer based on general knowledge
            - Format numbers and dates appropriately
            """
            
            user_prompt = f"Question: {question}\nData points: {str(data_points)}"
            
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt}
            ]
            
            eot = "<|eot_id|>"
            eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id
            
            tokenized_chat = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            inputs = self.tokenizer(
                tokenized_chat, 
                return_tensors='pt', 
                padding=True, 
                truncation=True
            )
            
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id=self.tokenizer.eos_token_id,
                max_new_tokens=1024,
                temperature=0.1,    #the probability distribution of the next word, 1 mean no change, below 1 more conservative, above 1 less predictable
                do_sample=True,  #influence the fundamental approach of model text generation, True for variability and creativity, false for more predictable and consistent text
                top_p = 0.1  #based on nucleus sampling, balance randomness and predictablity in text generation.

            )
            
            response = self.tokenizer.decode(outputs[0])
            response = response.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()[:-10]
            
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Xin lỗi, tôi đang gặp vấn đề trong việc xử lý câu trả lời."

class MultiAgentModel:
    """Main class that coordinates all agents"""
    def __init__(self, 
                 host: str,
                 user: str,
                 password: str,
                 database: str,
                 model_name: str = "phamhai/Llama-3.2-1B-Instruct-Frog"):
        # Initialize database connection
        db_config = DatabaseConfig(
            host=host,
            user=user,
            password=password,
            database=database
        )
        self.db_connection = DatabaseConnection(db_config)
        
        # Initialize agents
        self.sql_agent = SQLAgent("huyhoangt2201/llama-3.2-1b-sql_finetuned_billingual_3.0_merged")
        self.execute_agent = ExecuteQueryAgent(self.db_connection)
        self.llm_agent = LLMAgent("phamhai/Llama-3.2-1B-Instruct-Frog")
        
    def process_question(self, question: str) -> str:
        """Process user question through the entire pipeline"""
        try:
            # Step 1: Generate SQL query
            sql_query = self.sql_agent.generate_query(question)
            logger.info(f"Generated SQL query: {sql_query}")

            if sql_query.startswith('SELECT') == False:
                return sql_query
            
            # Step 2: Execute query and get results
            query_results = self.execute_agent.execute_query(sql_query)
            logger.info(f"Query results: {query_results}")
            
            # Step 3: Generate natural language response
            final_response = self.llm_agent.generate_response(question, query_results)
            
            return final_response
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lý câu hỏi của bạn."

# Example usage
if __name__ == "__main__":
    # Database configuration
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "2787853",
        "database": "db3"
    }
    
    # Initialize the model
    chatbot = MultiAgentModel(**DB_CONFIG)
    
    # Example questions
    questions = [
        "Cho tôi biết những cải tiến nào tiết kiệm được nhiều giờ nhất?",
        "Liệt kê các cải tiến được thực hiện trong tháng 11 năm 2024",
        "xin chào, bạn có khỏe không",
        "list all improvement from dcd",
        "hello, who are you?"

    ]
    
    # Process each question
    for question in questions:
        print(f"\nQuestion: {question}")
        response = chatbot.process_question(question)
        print(f"Response: {response}")
