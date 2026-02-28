import numpy as np
import pandas as pd

import os
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.utils.logger import get_logger
logger = get_logger("data_preprocessing")

# fetch the data from data/raw folder
def fetch_data():
    try:

        data_path = os.path.join("data", 'raw')
        train_data = pd.read_csv(os.path.join(data_path, 'train.csv'))
        test_data = pd.read_csv(os.path.join(data_path, 'test.csv'))
        return train_data, test_data
    except FileNotFoundError as e:
        logger.error(f"Error: the file was not found. Error detail: {e}")
        raise
    except pd.errors.EmptyDataError:
        logger.error(f"Error: the file is empty. Error detail: {e}")
        raise
    except pd.errors.ParserError:
        logger.error(f"Error: could not parse the data. Error detail: {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while fetching data. Error detail: {e}")
        raise

# transform the data    
nltk.download('wordnet')
nltk.download('stopwords')

def lemmatization(text: str) -> str:
    try:
        lemmatizer= WordNetLemmatizer()
        text = text.split()
        text=[lemmatizer.lemmatize(y) for y in text]
        return " " .join(text)
    
    except Exception as e:
        logger.error(f"An error occurred durring lemmatization: {e}")
        raise

def remove_stop_words(text: str) -> str:
    try:
        stop_words = set(stopwords.words("english"))
        Text=[i for i in str(text).split() if i not in stop_words]
        return " ".join(Text)
    
    except Exception as e:
        logger.error(f" An error occurred during top word removal: {e}")
        raise
    
def removing_numbers(text: str) -> str:
    try:
        text = ''.join([i for i in text if not i.isdigit()])
        return text
    except Exception as e:
        logger.error(f"An error occurred during number removal: {e}")
        raise

def lower_case(text: str) -> str:
    try:
        text = text.split()
        text=[y.lower() for y in text]
        return " " .join(text)
    except Exception as e:
        logger.error(f"An error occurred during lower case conversion: {e}")
        raise

def removing_punctuations(text: str) -> str:
    ## Remove punctuations
    try:
        text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
        text = text.replace('؛',"", )
        
        ## remove extra whitespace
        text = re.sub('\s+', ' ', text)
        text =  " ".join(text.split())
        return text.strip()
    except Exception as e:
        logger.error(f"An error occurred during puncuation removal: {e}")
        raise

def removing_urls(text: str) -> str:
    try:
        url_pattern = re.compile(r'https?://\S+|www\.\S+')
        return url_pattern.sub(r'', text)
    except Exception as e:
        logger.error(f"An error occurred during URL removal: {e}")
        raise

def remove_small_sentences(df: pd.DataFrame) -> pd.DataFrame:
    try:
        for i in range(len(df)):
            if len(df.text.iloc[i].split()) < 3:
                df.text.iloc[i] = np.nan
        return df
    except Exception as e:
        logger.error(f"An error occurred during small sentence removal: {e}")
        raise

def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    try:
        df.content=df.content.apply(lambda content : lower_case(content))
        logger.info("Lower case conversion completed successfully.")
        df.content=df.content.apply(lambda content : remove_stop_words(content))
        logger.info("Stop word removal completed successfully.")
        df.content=df.content.apply(lambda content : removing_numbers(content))
        logger.info("Number removal completed successfully.")
        df.content=df.content.apply(lambda content : removing_urls(content))
        logger.info("URL removal completed successfully.")
        df.content=df.content.apply(lambda content : removing_punctuations(content))
        logger.info("Punctuation removal completed successfully.")  
        df.content=df.content.apply(lambda content : lemmatization(content))
        logger.info("Lemmatization completed successfully.")   
        return df
    except Exception as e:
        logger.error(f"An error occurred during text normalization: {e}")
        raise   

def main():
    try:

        train_data, test_data = fetch_data()
        train_processed_data = normalize_text(df = train_data)
        test_processed_data = normalize_text(df = test_data)
        logger.info("Data preprocessing completed successfully.")

        # store the data in data/interim folder
        processed_data_path = os.path.join("data", 'interim')
        os.makedirs(processed_data_path, exist_ok=True)

        train_processed_data.to_csv(os.path.join(processed_data_path, 'train_processed.csv'), index=False)
        test_processed_data.to_csv(os.path.join(processed_data_path, 'test_processed.csv'), index=False)

        logger.info("Processed data saved successfully in the data/processed folder.")

    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")    
        raise

if __name__ == "__main__":
    main()
