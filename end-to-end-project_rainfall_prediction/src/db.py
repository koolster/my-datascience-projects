# This file contains functions related to downloading the data from the database

# Imports
import sqlite3
import pandas as pd 


def read_data_from_sqlite(database_file, table_name):
    """Function to read the data from sqlite file

    Args:
        database_file (str): Path to the database file
        table_name (str): Name of the table

    Returns:
        pandas dataframe: Dataframe containing the extracted table
    """
    
    # Establish a connection to the database
    conn = sqlite3.connect(database_file)
    
    # Read the data into a dataframe using the SQL query
    original_df = pd.read_sql_query(f"SELECT * from {table_name}", conn)
    
    # Close the connection
    conn.close()
    
    return original_df
