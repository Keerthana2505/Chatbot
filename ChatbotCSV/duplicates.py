import pandas as pd
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# Function to check if two sentences have the same meaning
def are_meaning_same(sentence1, sentence2):
    # Tokenize and lemmatize sentences
    lemmatizer = WordNetLemmatizer()
    tokens1 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence1)]
    tokens2 = [lemmatizer.lemmatize(word.lower()) for word in word_tokenize(sentence2)]
    # Check if sentences share the same words and their meanings
    for word1 in tokens1:
        for word2 in tokens2:
            # If any of the words have the same meaning, return True
            if word1 == word2:
                return True
            synsets1 = wordnet.synsets(word1)
            synsets2 = wordnet.synsets(word2)
            if synsets1 and synsets2:
                if synsets1[0].wup_similarity(synsets2[0]) is not None and synsets1[0].wup_similarity(synsets2[0]) > 0.9:
                    return True
    return False
# Function to remove duplicate sentences with the same meaning
def remove_duplicate_sentences(data):
    removed_indices = set()
    for i in range(len(data)):
        if i in removed_indices:
            continue
        for j in range(i + 1, len(data)):
            if j in removed_indices:
                continue
            if are_meaning_same(data[i], data[j]):
                removed_indices.add(j)
    return [sentence for i, sentence in enumerate(data) if i not in removed_indices]
# Read CSV file into a DataFrame
df = pd.read_csv('datas.csv')
# Convert DataFrame column to a list of sentences
sentences = df['Sentence'].tolist()
# Remove duplicate sentences with the same meaning
cleaned_sentences = remove_duplicate_sentences(sentences)
# Write cleaned data back to CSV file
cleaned_df = pd.DataFrame({'Sentence': cleaned_sentences})
cleaned_df.to_csv('cleaned_datas.csv', index=False)
print("Duplicates removed and saved to cleaned_datas.csv")