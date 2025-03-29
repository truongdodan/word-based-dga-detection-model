import pandas as pd
import os

matsnu = pd.read_csv(r"D:\HOCTAP\Machine_Learning\dga_dataset\matsnu.csv")
pushdo = pd.read_csv(r"D:\HOCTAP\Machine_Learning\dga_dataset\pushdo.csv")
suppobox = pd.read_csv(r"D:\HOCTAP\Machine_Learning\dga_dataset\suppobox.csv")
mixed = pd.read_csv(r"D:\HOCTAP\Machine_Learning\dga_dataset\mixed.csv")
top_1m = pd.read_csv(r"D:\HOCTAP\Machine_Learning\dga_dataset\top1m.csv", header=None)

# Preprocess datas on matsnu dga file
matsnu.columns = ['isDGA', 'domain', 'subclass'] # Change column name from 'host' to 'domain'
matsnu = matsnu.reindex(columns=['isDGA', 'subclass', 'domain']) # Change column order

# Preprocess datas on pushdo dga file
pushdo.columns = ['isDGA', 'domain', 'subclass'] # Change column name from 'host' to 'domain'
pushdo = pushdo.reindex(columns=['isDGA', 'subclass', 'domain']) # Change column order

# Preprocess datas on suppobox dga file
suppobox.columns = ['isDGA', 'domain', 'subclass'] # Change column name from 'host' to 'domain'
suppobox = suppobox.reindex(columns=['isDGA', 'subclass', 'domain']) # Change column order

# Preprocess datas on mixed domain file
mixed.columns = ['isDGA', 'subclass', 'domain'] # Change column name from 'host' to 'domain'

# Preprocess datas on legit domain file
top_1m.columns = ['domain'] # Set first column's name to 'domain'
top_1m['isDGA'] = 'legit' # Add new column and set all value to 'legit'
top_1m['subclass'] = 'alexa' # Add new column and set all value to 'alexa'
top_1m = top_1m.reindex(columns=['isDGA', 'subclass', 'domain']) # Change column order


# Combine all suppobox to 1 file
suppobox2 = mixed.loc[mixed['subclass'] == 'suppobox'] # Get all the suppobox in mixed domain

final_suppobox = pd.concat([suppobox, suppobox2]) # Combine 2 suppobox dataset to 1
final_suppobox.reset_index(drop=True)

print(final_suppobox.shape[0])
final_suppobox.drop_duplicates(subset=['domain'], keep=False, inplace=True) # Remove duplicate domain name
print(final_suppobox.shape[0])

print('________________suppobox')
print(final_suppobox)


# Final matsnu dataset
print(matsnu.shape[0])
matsnu.drop_duplicates(subset=['domain'], keep=False, inplace=True) # Remove duplicate domain name
print(matsnu.shape[0])

final_matsnu = matsnu.head(13583)
print('_____________________matsnu')
print(final_matsnu)


# Final pushdo dataset
print(pushdo.shape[0])
pushdo.drop_duplicates(subset=['domain'], keep=False, inplace=True) # Remove duplicate domain name
print(pushdo.shape[0])

final_pushdo = pushdo.head(13583)
print('_______________pushdo')
print(final_pushdo)

# Final legit domain dataset
print(top_1m.shape[0])
top_1m.drop_duplicates(subset=['domain'], keep=False, inplace=True) # Remove duplicate domain name
top_1m = top_1m.head(70000)
print(top_1m.shape[0])

print('___________________legit')
print(top_1m)



# FINAL DATASET 
final_dataset = pd.concat([top_1m, final_matsnu, final_suppobox, final_pushdo]) # Combine dataset
final_dataset.drop('subclass', axis=1, inplace=True) # Drop 'subclass' column - cause we don't need it!
final_dataset.reset_index(drop=True)
final_dataset = final_dataset.reindex(columns=['domain', 'isDGA']) # Change column order

# Preprocess final dataset
final_dataset # remove domain name extension
final_dataset['isDGA'] = final_dataset['isDGA'].map({'legit': 0, 'dga': 1}) # change isDGA to integer


# Export final datase
print("_________________FINAL_____________________\n")
print(final_dataset)
final_dataset.to_csv(r'D:\HOCTAP\Machine_Learning\detect_word-based_dga\preprocessing_dataset.csv', index=None, header=True)




