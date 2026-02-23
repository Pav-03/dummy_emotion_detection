import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

import os

import logging

#logging configure

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler("data_ingestion.log")
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def load_params(params_path: str)-> float:
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
            test_size = params['data_ingestion']['test_size']
            random_state = params['data_ingestion']['random_state']
            logger.info(f"Parameters loaded successfully from {params_path}. Test size: {test_size}, Random state: {random_state}")
            return test_size, random_state
    except FileNotFoundError:
        logger.error(f"Error: the file {params_path} was not found.")
        raise
    except yaml.YAMLError as e:
        logger.error(f"Error: could not parse the yaml file {params_path}. Error detail: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while loading parameters from {params_path}. Error detail: {e}")
        raise

def read_data(url: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(url)
        return df
    except pd.errors.EmptyDataError:
        logger.error(f"Error: the url {url} does not contain any data.")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error: could not parse the data from {url}.")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading data from {url}. Error detail: {e}")
        raise


def split_data(df: pd.DataFrame, test_size: float, random_state: int)-> pd.DataFrame:
    try:
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)
        return train_data, test_data
    except ValueError as e:
        logger.error(f"Error: invalid value for test_size or random_state. Error detail: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while splitting the data. Error detail: {e}")
        raise    

def process_data(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.drop(columns = ['tweet_id'], inplace = True)
        final_df = df[df['sentiment'].isin(['happiness','sadness'])]
        final_df['sentiment'] = final_df['sentiment'].map({'happiness':1, 'sadness':0})
        return final_df
    
    except KeyError as e:
        logger.error(f"Error: the expected column is missing in the dataframe. Error detail: {e}")
        raise

    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise

def save_data(data_path: str, train_data: pd.DataFrame, test_data: pd.DataFrame) -> None:
    try:
        os.makedirs(data_path, exist_ok=True)
        train_data.to_csv(os.path.join(data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(data_path, 'test.csv'), index=False)
    except OSError as e:
        logger.error(f"Error creating directory {data_path}. Error detail: {e}")
        raise

    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise

    

def main():
    try:
        
        test_size, random_state = load_params("params.yaml")
        df = read_data('https://raw.githubusercontent.com/campusx-official/jupyter-masterclass/main/tweet_emotions.csv')
        final_df = process_data(df)
        train_data, test_data = split_data(final_df, test_size, random_state)
        data_path = os.path.join("data", 'raw')
        save_data(data_path, train_data, test_data)

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        raise

if __name__ =="__main__":
    main()