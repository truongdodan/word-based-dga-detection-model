import pandas as pd
import numpy as np
import re
from wordsegment import load, segment
import enchant

# Number of rows in dataset: 110749
dga_dataset = pd.read_csv(r'D:\HOCTAP\Machine_Learning\detect_word-based_dga\dataset\feature_extraction_dataset.csv')

# Prepare data in dictionary folder
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

def num_exist_in_dict(words, dictionary):   # Count number of word in a word list that exist in another word list
    return sum(1 for word in words if word in dictionary) 

""" dga_domain = dga_dataset[dga_dataset["isDGA"] == 1]

extracted_word_list = []
for domain_name in dga_domain['domain']:   # Extract words from all the dga domain name
    extracted_words = extract_word_from_domain_name(domain_name)
    extracted_word_list.extend(extracted_words)

with open("D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\dga_domain_name_words.txt", "w", encoding="utf-8") as file:  # Save all the extracted words from dga domain to txt file
    file.write("\n".join(extracted_word_list)) 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\english_dict.txt") as file: # Get words from "english_dict"
    english_dict = [line.strip() for line in file] 

with open(r"D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\dga_domain_name_words.txt") as file: # Get words from "dga_domain_name_words"
    dga_domain_name_words = [line.strip() for line in file]  

unique_dga_domain_name_words = set(dga_domain_name_words) # Remove duplicate words.

dga_words = [word for word in unique_dga_domain_name_words if word in english_dict]    # Filter words that are used by dga domain and is in english dict
private_words = [word for word in unique_dga_domain_name_words if word not in english_dict]    # Filter wordsthat are used by dga domain and is not in english dict

with open("D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\dga_dict.txt", "w", encoding="utf-8") as file:  # Save to dga_dict
    file.write("\n".join(dga_words))

with open("D:\HOCTAP\Machine_Learning\detect_word-based_dga\dictionary\private_dict.txt", "w", encoding="utf-8") as file:  # Save to private_dict
    file.write("\n".join(private_words)) """

""" # Get dictionaries
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
    private_dict = [line.strip() for line in file]  """



""" # FEATURE 1 - 6
domain_length_array = np.array([]) # F1
domain_total_ASCII = np.array([]) # F2
domain_vowel_count = np.array([]) # F3
vowels = {"a", "i", "u", "e", "o", "A", "I", "U", "E", "O"} # F5
digit_and_hyphen_count = np.array([]) # F5

for domain_name in dga_dataset["domain"]:
    # F1: Length of domain name.
    domain_length_array = np.append(domain_length_array, len(domain_name))

    # F2: Total value in ASCII code of all characters.
    ascii_sum = sum(ord(char) for char in domain_name)
    domain_total_ASCII = np.append(domain_total_ASCII, ascii_sum)

    # F3: Number of Vowels of domain name.
    vowels_count = sum(1 for char in domain_name if char in vowels)
    domain_vowel_count = np.append(domain_vowel_count, vowels_count)

    # F5: Number of digits and characters '-' in domain name.
    di_and_hi = sum(char.isdigit() for char in domain_name) + domain_name.count("-")
    digit_and_hyphen_count = np.append(digit_and_hyphen_count, di_and_hi)

dga_dataset.insert(0, "length", domain_length_array.astype(int)) # F1
dga_dataset.insert(1, "total_ASCII", domain_total_ASCII.astype(int)) # F2
dga_dataset.insert(2, "vowel_count", domain_vowel_count.astype(int)) # F3
dga_dataset.insert(4, "digit_and_hyphen_count", digit_and_hyphen_count.astype(int)) # F5

# F4: Vowel distribution of domain name with formula: F3/F1.
dga_dataset.insert(3, "vowel_distribution", dga_dataset["vowel_count"] / dga_dataset["length"])

# F6: Digit and character '-' distribution with formula: F5/F1.
dga_dataset.insert(5, "digit_and_hyphen_distribution", dga_dataset["digit_and_hyphen_count"] / dga_dataset["length"])

# Export dataset with 6 feature
dga_dataset.to_csv(r'D:\HOCTAP\Machine_Learning\detect_word-based_dga\feature_extraction_dataset.csv', index=None, header=True) """




# FEATURE 7 - 16
""" word_norm = [] # F7
word_dga = [] # F8
noun_count = [] # F9
verb_count = [] # F10
adj_count = [] # F11
private_count = [] # F12
for domain_name in dga_dataset["domain"]:
    extracted_words = extract_word_from_domain_name(domain_name)

    # F7: Word extract from domain name and exist in english_dict - word_norm.
    word_norm_count = sum(1 for word in extracted_words if word in english_dict)
    word_norm.append(word_norm_count)

    # F8: Word extract from domain name and exist in dga_dict - word_dga.
    word_dga_count = sum(1 for word in extracted_words if word in dga_dict)
    word_dga.append(word_dga_count)

    # F9: Word extract from domain name and exist in noun_dict - noun_count.
    word_noun_count = sum(1 for word in extracted_words if word in noun_dict)
    noun_count.append(word_noun_count)

    # F10: Word extract from domain name and exist in verb_dict - verb_count.
    word_verb_count = sum(1 for word in extracted_words if word in verb_dict)
    verb_count.append(word_verb_count)

    # F11: Word extract from domain name and exist in adj_dict - adj_count.
    word_adj_count = sum(1 for word in extracted_words if word in adj_dict)
    adj_count.append(word_adj_count)

    # F12: Word extract from domain name and exist in private_dict - private_count.
    word_private_count = sum(1 for word in extracted_words if word in private_dict)
    private_count.append(word_private_count)

dga_dataset.insert(7, "word_norm", word_norm) # F7
dga_dataset.insert(8, "word_dga", word_dga) # F8
dga_dataset.insert(9, "noun_count", noun_count) # F9
dga_dataset.insert(10, "verb_count", verb_count) # F10
dga_dataset.insert(11, "adj_count", adj_count) # F11
dga_dataset.insert(12, "private_count", private_count) # F12 

# F13: Ratio between word_dga and word_norm = word_dga/word_norm.
dga_dataset.insert(13, "dga_and_norm_ratio", dga_dataset["word_dga"] / dga_dataset["word_norm"])
dga_dataset["dga_and_norm_ratio"].fillna(0, inplace=True) """

""" longest_word_lens = [] # F14
shortest_word_lens = [] # F15
ratio = [] # F15
for domain_name in dga_dataset["domain"]:
    extracted_words = extract_word_from_domain_name(domain_name)

    if extracted_words:
        # F14: Length of longest word in domain.
        longest_word_lens.append(len(max(extracted_words, key=len)))

        # F15: Length of shortest word in domain.
        shortest_word_lens.append(len(min(extracted_words, key=len)))
    else: 
        longest_word_lens.append(0)    # F14
        shortest_word_lens.append(0)   # F15

    # F16: Ratio: len(word(d))/len(d).
    total_words_len = sum(len(word) for word in extracted_words)
    domain_without_extension = domain_name.rsplit('.', 1)[0]
    ratio.append(total_words_len/len(domain_without_extension))


dga_dataset.insert(14, "longest_word_len", longest_word_lens) # F14
dga_dataset.insert(15, "shortest_word_len", shortest_word_lens) # F15
dga_dataset.insert(16, "words_ratio", ratio) # F16 """



dga_dataset.drop('domain', axis=1, inplace=True)    # Remove 'domain' column
dga_dataset.to_csv(r'D:\HOCTAP\Machine_Learning\detect_word-based_dga\final_dataset.csv', index=None, header=True) # Export dataset                            
with pd.option_context('display.max_rows', None, 'display.max_columns', None, "display.expand_frame_repr", False):
    print(dga_dataset.tail(10))

