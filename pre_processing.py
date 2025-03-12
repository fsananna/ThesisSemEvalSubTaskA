import pandas as pd
import re
import nltk
import unicodedata
import string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
from wordsegment import load, segment

# Initialize NLTK resources
nltk.download(['punkt', 'wordnet', 'stopwords', 'punkt_tab'])
load()

def preprocess_text(text):
    """Text preprocessing pipeline"""
    if not isinstance(text, str):
        return ""
    
    # Basic cleaning
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'http\S+|www\S+|@\w+|#(\w+)', '', text)  # Remove URLs, mentions
    text = text.lower()  # Lowercase
    
    # Word segmentation for hashtags
    text = re.sub(r'#(\w+)', lambda m: ' '.join(segment(m.group(1))), text)
    
    # Remove punctuation/numbers
    text = text.translate(str.maketrans('', '', string.punctuation + '0123456789'))
    
    # Tokenization and lemmatization
    words = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    
    return ' '.join(words)

def process_dataset(input_path, output_path):
    """Process TSV file with error handling"""
    try:
        df = pd.read_csv(input_path, delimiter='\t', encoding='utf-8')
        if 'sentence' not in df.columns:
            raise ValueError("Dataset missing 'sentence' column")
            
        df['processed_sentence'] = df['sentence'].apply(preprocess_text)
        df.to_csv(output_path, sep='\t', index=False)
        print(f"Success! Processed data saved to {output_path}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

if __name__ == "__main__":
    input_path = './train.tsv'          # Update this path
    output_path = './train_processed.tsv'  # Update this path
    process_dataset(input_path, output_path)