# This file contains functions related to cleaning the dataframe

import pandas as pd
import numpy as np

def add_date_columns(df, date_column):
    """This function adds Year, Month and Week information 
    based on the date column

    Args:
        df (pandas DataFrame): Input DataFrame
        date_column (str): Name of the date columnn

    Returns:
        pandas DataFrame: Output DataFrame
    """
    
    # Convert date column to datetime type
    df[date_column] = pd.to_datetime(df[date_column])

    # Add year, month, and week columns
    df['Year'] = df[date_column].dt.year
    df['Month'] = df[date_column].dt.month
    df['Week'] = df[date_column].dt.isocalendar().week
    
    #drop date_column 
    df.drop(columns=[date_column], inplace=True, axis=1)
    
    # Return the updated DataFrame
    return df

def clean_df(df):
    """This function performs data cleaning steps before the ML model

    Args:
        df (pandas DataFrame): Input DataFrame

    Returns:
        Output: Output DataFrame
    """
    
    # Handle Date column
    df = add_date_columns(df, 'Date')

    # Handle Sunshine column
    df['Sunshine'] = np.abs(df['Sunshine'])

    # Set Wind Direction to 'No Directions' when they are NA
    df["WindGustDir"] = df["WindGustDir"].fillna("NoDirection")
    df["WindDir9am"] = df["WindDir9am"].fillna("NoDirection")
    df["WindDir3pm"] = df["WindDir3pm"].fillna("NoDirection")

    # Set wind speeds to 0 when they are NA
    df["WindGustSpeed"] = df["WindGustSpeed"].fillna(0)
    df["WindSpeed9am"] = df["WindSpeed9am"].fillna(0)
    df["WindSpeed3pm"] = df["WindSpeed3pm"].fillna(0)

    # Convert Pressure columns to Upper
    df['Pressure9am'] = df['Pressure9am'].str.upper()
    df['Pressure3pm'] = df['Pressure3pm'].str.upper()

    # Fill the Pressure columns wtih from the other pressure measurements if there
    df.loc[df['Pressure9am'].isna(), 'Pressure9am'] = df.loc[df['Pressure9am'].isna(), 'Pressure3pm']
    df.loc[df['Pressure3pm'].isna(), 'Pressure3pm'] = df.loc[df['Pressure3pm'].isna(), 'Pressure9am']

    # Fill Other NAs with "MED"
    df['Pressure9am'] = df['Pressure9am'].fillna('MED')
    df['Pressure3pm'] = df['Pressure3pm'].fillna('MED')

    # Drop 'ColourOfBoats' 
    df = df.drop('ColourOfBoats', axis=1)

    # Update RainToday column based on Rainfall
    df['RainToday'] =  df['Rainfall'].apply(lambda row: 'Yes' if row > 1 else 'No')
    
    # Map target column to numeric (For XGboost to work)
    df['RainTomorrow'] = df['RainTomorrow'].map({'Yes' : 1, 'No': 0})

    return df



    