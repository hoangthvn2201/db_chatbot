def convert_mysql_to_mssql(query: str) -> str:
    """
    Comprehensive converter from MySQL to Microsoft SQL Server query syntax
    
    Args:
        query (str): Original MySQL query
    
    Returns:
        str: Converted Microsoft SQL Server query
    """
    # Normalize the query
    query = query.strip()
    
    # 1. Handle LIMIT and OFFSET
    query = handle_limit_offset(query)
    
    # 2. Replace MySQL-specific functions
    query = replace_mysql_functions(query)
    
    # 3. Handle boolean and boolean-like operations
    query = convert_boolean_operations(query)
    
    # 4. Handle LIKE case sensitivity
    query = handle_like_case_sensitivity(query)
    
    # 5. Replace MySQL-specific date and time functions
    query = convert_date_time_functions(query)
    
    # 6. Handle column and table name conflicts
    query = handle_reserved_keywords(query)
    
    # 7. Replace GROUP_CONCAT
    query = replace_group_concat(query)
    
    # 8. Handle REGEXP
    query = convert_regexp(query)
    
    return query.strip()

def handle_limit_offset(query: str) -> str:
    """
    Convert LIMIT and OFFSET to TOP and window functions
    """
    # Handle queries with both LIMIT and OFFSET
    offset_match = re.search(r'LIMIT\s+(\d+)\s*,\s*(\d+)', query, re.IGNORECASE)
    if offset_match:
        offset, limit = offset_match.groups()
        # Replace with window function for pagination
        query = re.sub(
            r'LIMIT\s+\d+\s*,\s*\d+', 
            f'OFFSET {offset} ROWS FETCH NEXT {limit} ROWS ONLY', 
            query, 
            flags=re.IGNORECASE
        )
    
    # Handle simple LIMIT
    limit_match = re.search(r'LIMIT\s+(\d+)', query, re.IGNORECASE)
    if limit_match and 'ORDER BY' in query.upper():
        limit_value = limit_match.group(1)
        query = re.sub(r'LIMIT\s+\d+', f'TOP {limit_value}', query, flags=re.IGNORECASE)
    elif limit_match:
        limit_value = limit_match.group(1)
        # If no ORDER BY, insert TOP at the right place
        select_match = re.search(r'^(SELECT)\s', query, re.IGNORECASE)
        if select_match:
            query = query.replace(select_match.group(1), f'{select_match.group(1)} TOP {limit_value}', 1)
        query = re.sub(r'LIMIT\s+\d+', '', query, flags=re.IGNORECASE)
    
    return query

def replace_mysql_functions(query: str) -> str:
    """
    Replace MySQL-specific functions with their MSSQL equivalents
    """
    # Replace MySQL functions
    replacements = {
        r'\bNOW\(\)': 'GETDATE()',
        r'\bCURDATE\(\)': 'CAST(GETDATE() AS DATE)',
        r'\bUUID\(\)': 'NEWID()',
        r'\bCONCAT\(': 'CONCAT(',
        r'\bIF\(': 'IIF(',
        r'\bREPEAT\(([^,]+),\s*([^)]+)\)': r'REPLICATE(\1, \2)'
    }
    
    for pattern, replacement in replacements.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    return query

def convert_boolean_operations(query: str) -> str:
    """
    Convert MySQL boolean operations to MSSQL syntax
    """
    # Replace boolean keywords
    query = query.replace(' AND ', ' AND ')
    query = query.replace(' OR ', ' OR ')
    query = query.replace('TRUE', '1')
    query = query.replace('FALSE', '0')
    
    return query

def handle_like_case_sensitivity(query: str) -> str:
    """
    Handle LIKE case sensitivity
    """
    # Add COLLATE for case-insensitive LIKE
    if ' LIKE ' in query.upper():
        query = re.sub(
            r'(\w+)\s+LIKE\s+(\S+)', 
            r'\1 LIKE \2 COLLATE SQL_Latin1_General_CP1_CI_AS', 
            query, 
            flags=re.IGNORECASE
        )
    
    return query

def convert_date_time_functions(query: str) -> str:
    """
    Convert MySQL date and time functions to MSSQL equivalents
    """
    # Date and time function replacements
    replacements = {
        r'\bDATE_FORMAT\(([^,]+),\s*\'([^\']+)\'\)': r'FORMAT(\1, \2)',
        r'\bDAYOFWEEK\(([^)]+)\)': r'DATEPART(dw, \1)',
        r'\bYEAR\(([^)]+)\)': r'YEAR(\1)',
        r'\bMONTH\(([^)]+)\)': r'MONTH(\1)',
        r'\bDAY\(([^)]+)\)': r'DAY(\1)',
        r'\bDATE_ADD\(([^,]+),\s*INTERVAL\s+(\d+)\s*(\w+)\)': convert_date_add
    }
    
    for pattern, replacement in replacements.items():
        query = re.sub(pattern, replacement, query, flags=re.IGNORECASE)
    
    return query

def convert_date_add(match):
    """
    Convert MySQL DATE_ADD to MSSQL equivalent
    """
    date, interval, unit = match.groups()
    unit = unit.upper()
    
    units_map = {
        'DAY': 'day',
        'MONTH': 'month',
        'YEAR': 'year',
        'HOUR': 'hour',
        'MINUTE': 'minute',
        'SECOND': 'second'
    }
    
    mssql_unit = units_map.get(unit, unit.lower())
    return f"DATEADD({mssql_unit}, {interval}, {date})"

def handle_reserved_keywords(query: str) -> str:
    """
    Escape reserved keywords and problematic column names
    """
    # Add square brackets around reserved words and potential problematic names
    reserved_words = ['user', 'order', 'group', 'select', 'key', 'table']
    
    for word in reserved_words:
        # Use word boundaries to avoid partial matches
        query = re.sub(r'\b' + word + r'\b', f'[{word}]', query, flags=re.IGNORECASE)
    
    return query

def replace_group_concat(query: str) -> str:
    """
    Replace GROUP_CONCAT with STRING_AGG in MSSQL
    """
    # Basic GROUP_CONCAT replacement
    group_concat_match = re.search(r'GROUP_CONCAT\(([^)]+)\)', query, re.IGNORECASE)
    if group_concat_match:
        concat_content = group_concat_match.group(1)
        # Basic conversion, might need more sophisticated parsing for complex cases
        query = re.sub(
            r'GROUP_CONCAT\([^)]+\)', 
            f'STRING_AGG({concat_content}, \',\')', 
            query, 
            flags=re.IGNORECASE
        )
    
    return query

def convert_regexp(query: str) -> str:
    """
    Convert REGEXP to LIKE with LIKE multiple patterns
    """
    # Replace REGEXP with more complex LIKE patterns
    query = re.sub(r'\s+REGEXP\s+', ' LIKE ', query, flags=re.IGNORECASE)
    
    return query
