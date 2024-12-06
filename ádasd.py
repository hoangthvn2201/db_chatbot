def convert_sql_query(query: str) -> str:
    """
    Convert SQL query to be compatible with Microsoft SQL Server syntax
    
    Args:
        query (str): Original SQL query
    
    Returns:
        str: Converted SQL query compatible with Microsoft SQL Server
    """
    # Replace LIMIT with TOP
    if 'LIMIT' in query.upper():
        # Extract the LIMIT value
        limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
        if limit_match:
            limit_value = limit_match.group(1)
            # Replace LIMIT with TOP
            query = re.sub(r'LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
            
            # Check if SELECT is the first word
            if query.upper().startswith('SELECT'):
                query = query.replace('SELECT', f'SELECT TOP {limit_value}', 1)
            else:
                # If SELECT is not at the start, we'll add TOP manually
                query = f'SELECT TOP {limit_value} {query[6:]}'
    
    # Modify column names
    query = query.replace('ImprovementName', 'ImprovementContent')
    query = query.replace('TimeSaving', 'TotalTimeSaved')
    
    return query
