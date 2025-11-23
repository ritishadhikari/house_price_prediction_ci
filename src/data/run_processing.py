import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger=logging.getLogger("data-processor")

def load_data(file_path) -> pd.DataFrame:
    """
    Load data from a CSV file
    """
    logger.info(f"Loading data from {file_path}")
    df=pd.read_csv(file_path)
    logger.info(f"Source Data has shape: {df.shape}")
    return df

def clean_data(df:pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by handling missing values and outliers
    """
    logger.info("Cleaning Dataset")

    # Make a copy to avoid modifying the original dataframe
    df_cleaned=df.copy()

    # Handle Missing Value
    for column in df_cleaned.columns:
        missing_count=df_cleaned[column].isnull().sum()
        if missing_count>0:
            logger.info(f"Found {missing_count} missing values in {column}")

            # For numeric columns, fill with the median
            if pd.api.types.is_numeric_dtype(df_cleaned[column]):
                median_value=df_cleaned[column].median()
                df_cleaned[column]=df_cleaned[column].fillna(median_value)
                logger.info(f"Filled Missing Values in {column} with median: {median_value}")
            # For Categorical columns, fill with mode
            else:
                mode_value=df_cleaned[column].mode()[0]
                df_cleaned[column]=df_cleaned[column].fillna(mode_value)
                logger.info(f"Filled Missing Values in {column} with mode: {mode_value}")
    
    # Handling outliers in price(target variable)
    # Using IQR method to identify outliers
    Q1=df_cleaned['price'].quantile(0.25)
    Q3=df_cleaned['price'].quantile(0.75)
    IQR=Q3-Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=df_cleaned[
        (df_cleaned['price']<lower_bound) |
        (df_cleaned['price']>upper_bound)
    ]
    if not outliers.empty:
        logger.info(f"Found {len(outliers)} outliers in price column")
        df_cleaned=df_cleaned.loc[
            (df_cleaned['price']>=lower_bound) &
            (df_cleaned['price']<=upper_bound),
            :
        ]
        logger.info(f"Removed outliers. New Dataset shape: {df_cleaned.shape}")
    return df_cleaned
        
def process_data(input_file,output_file):
    """
    Full Data Processing Pipeline
    """
    # Create output directory if it does not exists
    output_path=Path(output_file).parent  # WindowsPath('data/raw')
    output_path.mkdir(parents=True,exist_ok=True)

    # Load Data
    df=load_data(file_path=input_file)

    # Clean Data
    df_cleaned=clean_data(df=df)

    # Save Processed Data
    df_cleaned.to_csv(output_file,index=False)
    logger.info(f"Saved processed data to {output_file}")
    return df_cleaned
    
if __name__=="__main__":
    process_data(
        input_file="data/raw/house_data.csv",
        output_file="data/processed/cleaned_house_data.csv"
    )
