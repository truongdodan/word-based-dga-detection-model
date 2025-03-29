import pandas as pd
import numpy as np
import re
from wordsegment import load, segment
import enchant
import math
import joblib
from flask import Flask, request, render_template
from urllib.parse import urlparse

# Get dictionary
with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\english_dict.txt") as file: # Get english_dict.txt
    english_dict = [line.strip() for line in file] 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\noun_dict.txt") as file: # Get noun_dict.txt
    noun_dict = [line.strip() for line in file] 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\verb_dict.txt") as file: # Get verb_dict.txt
    verb_dict = [line.strip() for line in file] 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\adj_dict.txt") as file: # Get adj_dict.txt
    adj_dict = [line.strip() for line in file] 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\dga_dict.txt") as file: # Get dga_dict.txt
    dga_dict = [line.strip() for line in file] 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\private_dict.txt") as file: # Get private_dict.txt
    private_dict = [line.strip() for line in file]

# Extract works from domain name
load()
dict = enchant.Dict("en_US")
def extract_word_from_domain_name(domain):  # Extract words from domain name
    # Remove extension
    domain = domain.rsplit('.', 1)[0]

    # Remove special characters and numbers
    domain = re.sub('[^A-Za-z]+', '', domain)

    # Extract words
    word_list = segment(domain)

    # Extract only valid word
    word_list = [word for word in word_list if dict.check(word)] 
    
    return word_list

# Extract features from domain name
def feature_extraction(domain_name):
    vowels = {"a", "i", "u", "e", "o", "A", "I", "U", "E", "O"} # F5
    extracted_words = extract_word_from_domain_name(domain_name)
    f13 = sum(1 for word in extracted_words if word in dga_dict) / sum(1 for word in extracted_words if word in english_dict) # F13
    															

    features = {
            "length": len(domain_name), # F1
            "total_ASCII": sum(ord(char) for char in domain_name), # F2
            "vowel_count": sum(1 for char in domain_name if char in vowels), # F3 
            "vowel_distribution": sum(1 for char in domain_name if char in vowels) / len(domain_name), # F4
            "digit_and_hyphen_distribution": sum(char.isdigit() for char in domain_name) + domain_name.count("-"), # F5
            "digit_and_hyphen_count": sum(char.isdigit() for char in domain_name) + domain_name.count("-") / len(domain_name), # F6
            "word_norm": sum(1 for word in extracted_words if word in english_dict), # F7
            "word_dga": sum(1 for word in extracted_words if word in dga_dict), # F8
            "noun_count": sum(1 for word in extracted_words if word in noun_dict), # F9
            "verb_count": sum(1 for word in extracted_words if word in verb_dict), # F10
            "adj_count": sum(1 for word in extracted_words if word in adj_dict), # F11
            "private_count": sum(1 for word in extracted_words if word in private_dict), # F12
            "dga_and_norm_ratio": f13 if not math.isnan(f13) else 0, # F13
            "longest_word_len": len(max(extracted_words, key=len)) if extracted_words else 0, # F14
            "shortest_word_len": len(min(extracted_words, key=len)) if extracted_words else 0, # F15
            "words_ratio": sum(len(word) for word in extracted_words) / len(domain_name.rsplit('.', 1)[0]), # F16
        }
    return pd.DataFrame([features])

# Load model 
model = joblib.load(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\code\dga_detecting_model.pkl")

# Extract domain name from URL
def extract_domain(url):
    parsed_url = urlparse(url)
    domain_name = parsed_url.netloc if parsed_url.netloc else parsed_url.path
    return domain_name

# Initialize Flask app
app = Flask(__name__)

# Setup routes
@app.route('/')    # Home page route
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])    # Predict route
def predict_domain():
    # Get url from Form
    url = request.form['url']
    domain_name = extract_domain(url)

    # Extract features
    features = feature_extraction(domain_name)

    # Predict
    prediction = model.predict(features)[0]
    result = "DGA" if prediction == 1 else "Non-DGA"

    return render_template('index.html', prediction_result=result, prediction=prediction, domain_name=domain_name)

if __name__ == "__main__":
    app.run(debug=True)

""" with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.expand_frame_repr", False):
    print(feature_extraction("thisisadomainname.vn")) """