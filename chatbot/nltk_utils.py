import nltk
import spacy
import numpy as np
#il faut telecharger punk
nltk.download('punkt')
nlp = spacy.load("fr_core_news_sm")
def tokenize(sentence):
    return nltk.word_tokenize(sentence)
""""
#def stem(sentence):
    doc = nlp(sentence)
    lemmatized_tokens = [token.lemma_ for token in doc]
    lemmatized_sentence = ' '.join(lemmatized_tokens)
    return lemmatized_tokens

#def lemmatize_words(words):
#    lemmatized_forms = [token.lemma_ for token in nlp(' '.join(words))]
#    return lemmatized_forms
"""""
def lemmatize_and_lower(tokens):
    lemmatized_lower_forms = [token.lemma_.lower() for token in nlp(' '.join(tokens))]
    return lemmatized_lower_forms


def lemmatize_and_lower_single_word(word):
    return nlp(word)[0].lemma_.lower()

def bag_of_word(tokenized_sentence,all_words):
    tokenized_sentence=[lemmatize_and_lower_single_word(words) for words in tokenized_sentence]
    bag=np.zeros(len(all_words),dtype=np.float32)
    for idx , words in enumerate(all_words):
        if words in tokenized_sentence:
            bag[idx]=1.0
    
    return bag

