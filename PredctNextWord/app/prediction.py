import random

import numpy as np
import pandas as pd

from nltk.tokenize import RegexpTokenizer
from tensorflow.keras.models import load_model
from django.conf import settings


print(settings.BASE_DIR)

text_df = pd.read_csv(f"{settings.BASE_DIR}/fake_or_real_news.csv")
text = list(text_df.text.values)
joined_text = " ".join(text)
partial_text = joined_text[:1100000]

tokenizer = RegexpTokenizer(r"\w+")
tokens = tokenizer.tokenize(partial_text.lower())

unique_tokens = np.unique(tokens)
unique_token_index = {token: idx for idx, token in enumerate(unique_tokens)}

n_words = 10
input_words = []
next_words = []
for i in range(len(tokens) - n_words):
    input_words.append(tokens[i:i + n_words])
    next_words.append(tokens[i + n_words])

model = load_model(f"{settings.BASE_DIR}/model.h5")

def predict_next_word(input_text, n_best):
    input_text = input_text.lower()
    x = np.zeros((1, n_words, len(unique_tokens)))
    for i, word in enumerate(input_text.split()):
        x[0, i, unique_token_index[word]] = 1
    predictions = model.predict(x)[0]
    return np.argpartition(predictions, -n_best)[-n_best:]

def generate_text(input_text, text_length, creativity=3):
    word_sequence = input_text.split()
    current = 0
    for _ in range(text_length):
        sub_sequence = " ".join(tokenizer.tokenize(" ".join(word_sequence).lower())[current:current+n_words])
        try:
            choice = unique_tokens[random.choice(predict_next_word(sub_sequence, creativity))]
        except:
            choice = random.choice(unique_tokens)
        word_sequence.append(choice)
        current += 1
    return " ".join(word_sequence)