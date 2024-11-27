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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
    
    def __init__(self, model_name: str = ''):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        self.context = """You are an SQL query assistant. Based on schema, generate an SQL query to retrieve the relevant information for the user. If the user’s question is unrelated to the table, respond naturally in user's language.
        Schema:
        +Table Author, columns=[AuthorId: int, AuthorName: nvarchar(255), DepartmentId int, GroupDCId int]
        +Table Department, columns=[DepartmentId: int, DepartmentName: nvarchar(255)]
        +Table GroupDC, columns=[GroupDCId: int, DepartmentId: int, GroupDCName nvarchar(255)]
        +Table Job, columns=[JobId: int, JobName: nvarchar(255)]
        +Table Tool, columns=[ToolId: int, ToolName: nvarchar(255), ToolDescription: text]
        +Table Jidouka, columns=[JidoukaId: bigint, ProductApply: nvarchar(255), ImprovementName: nvarchar(255), SoftwareUsing: nvarchar(255), Description: nvarchar(255), Video: text, DetailDocument: text, TotalJobApplied: int, TotalTimeSaved: int, DateCreate: datetime, JobId: int, AuthorId: int, DepartmentId: int, GroupDCId: int]
        +Table JidoukaTool, columns=[JidoukaId: bigint, ToolId: int]
        +Primary_keys=[Author.AuthorId, Department.DepartmentId, GroupDC.GroupDCId, Job.JobId, Tool.ToolId, Jidouka.JidoukaId]
        +Foreign_keys=[GroupDC.DepartmentId=Department.DepartmentId, Jidouka.JobId=Job.JobId, Jidouka.AuthorId=Author.AuthorId, Jidouka.DepartmentId=Department.DepartmentId, Jidouka.GroupDCId=GroupDC.GroupDCId, JidoukaTool.JidoukaId=Jidouka.JidoukaId, JidoukaTool.ToolId=Tool.ToolId, Author.DepartmentId=Department.DepartmentId, Author.GroupDCId=GroupDC.GroupDCId]"""

    def generate_query(self, question: str) -> str:
        try:
            messages = [
                {'role': 'system', 'content': self.context},
                {'role': 'user', 'content': question}
            ]

            eot = '<|eot_id|>'
            eot_id = self.tokenizer.convert_tokens_to_ids(eot)
            self.tokenizer.pad_token = eot
            self.tokenizer.pad_token_id = eot_id 

            chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(
                chat,
                padding=True,
                truncation=True,
                return_tensors='pt').to(DEVICE)
            
            outputs = self.model.generate(
                inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                pad_token_id = self.tokenizer.eos_token_id,
                max_new_tokens=512,
                temperature = 0.1,
                do_sample = True,
                top_p = 0.1
            ).to(DEVICE)

            response = self.tokenizer.decode(outputs[0])
            response = response.split('<|start_header_id|>assistant<|end_header_id|>')[1].strip()[:-10]

            return response
        
        except Exception as e:
            return "Xin lỗi, tôi đang gặp vấn đề trong việc xử lí câu hỏi của bạn"
        

class ExecuteQuery:
    """Execute SQL queries and handle responses"""
    def __init__(self, db_connection: DatabaseConnection):
        self.db_connection = db_connection
        
    def is_valid_sql_query(self, query: str) -> bool:
        """Check if the string is a valid SQL query"""
        # if query.startswith("NOT_SQL_QUERY:"):
        #     return False
            
        # sql_keywords = ["SELECT", "FROM", "WHERE", "GROUP BY", "ORDER BY", "HAVING", "JOIN"]
        # query_upper = query.upper()
        # return any(keyword in query_upper for keyword in sql_keywords)
        if query.upper().startswith('SELECT'):
            return True 
        return False
    
    def sanitize_query(self, query: str) -> str:
        """Basic SQL injection prevention"""
        # Remove comments
        query = re.sub(r'/\*.*?\*/', '', query)
        query = re.sub(r'--.*$', '', query)
        query = re.sub('ImprovementName', 'ImprovementContent', query)
        
        # Remove multiple semicolons
        query = query.replace(';', '')
        
        return query
    
    def execute_query(self, query: str) -> Union[List[Dict[str, Any]], str]:
        # """Execute SQL query and return results"""
        # if not self.is_valid_sql_query(query):
        #     if query.startswith("NOT_SQL_QUERY:"):
        #         return query[13:].strip()
        #     return []
            
        try:
            sanitized_query = self.sanitize_query(query)
            with self.db_connection.get_connection() as connection:
                df = pd.read_sql_query(sanitized_query, connection)
                return df.to_dict('records')
        except Exception as e:
            return []
class UniModel:

    def __init__(self,
                 host: str,
                 user: str,
                 password: str,
                 database: str,
                 model_name: str = ''):
        
        db_config = DatabaseConfig(
            host=host,
            user=user,
            password=password,
            database=database
        )

        self.db_connection = DatabaseConnection(db_config)
        self.sql_agent = SQLAgent("")
        self.execute_agent = ExecuteQuery(self.db_connection)
    
    def process_question(self, question: str) -> str: 
        try: 
            sql_query = self.sql_agent.generate_query(question)

            if not sql_query.startswith('SELECT'):
                return sql_query 

            query_results = self.execute_agent.execute_query(sql_query)

            return query_results
        except Exception as e:
            return "Xin lỗi, đã có lỗi xảy ra trong quá trình xử lí câu hỏi của bạn."
        
if __name__ == "__main__":
    # Database configuration
    DB_CONFIG = {
        "host": "localhost",
        "user": "root",
        "password": "2787853",
        "database": "db3"
    }
    
    # Initialize the model
    chatbot = UniModel(**DB_CONFIG)

    question = ''

    response = chatbot.process_question(question)
