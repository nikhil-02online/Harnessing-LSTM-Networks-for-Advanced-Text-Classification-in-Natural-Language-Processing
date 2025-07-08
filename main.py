import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import re
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from sklearn.metrics import confusion_matrix

nltk.download('punkt')
nltk.download('stopwords')

st.title("📩 Spam/Ham Classifier using LSTM & Doc2Vec")

@st.cache_data
def load_data():
    df = pd.read_csv('./SPAM text message 20170820 - Data.csv', delimiter=',', encoding='latin-1')
    df = df[['Category', 'Message']]
    df = df.dropna()
    df.rename(columns={'Message': 'Message'}, inplace=True)
    df.index = range(len(df))
    return df

df = load_data()

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'\|\|\|', r' ', text)
    text = re.sub(r'http\S+', r'<URL>', text)
    text = text.lower()
    return text

df['Message'] = df['Message'].apply(clean_text)

def tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            tokens.append(word.lower())
    return tokens

if 'd2v_model' not in st.session_state:
    train_tagged = df.apply(lambda r: TaggedDocument(words=tokenize_text(r['Message']), tags=[r.Category]), axis=1)

    def train_doc2vec(train_tagged):
        d2v_model = Doc2Vec(dm=1, dm_mean=1, vector_size=20, window=8, min_count=1, workers=1, alpha=0.065, min_alpha=0.065)
        d2v_model.build_vocab(train_tagged.values)
        
        for epoch in range(30):
            d2v_model.train(train_tagged.values, total_examples=len(train_tagged.values), epochs=1)
            d2v_model.alpha -= 0.002
            d2v_model.min_alpha = d2v_model.alpha
            
        return d2v_model
    
    st.session_state.d2v_model = train_doc2vec(train_tagged)

d2v_model = st.session_state.d2v_model  

@st.cache_data
def preprocess_data(df):
    tokenizer = Tokenizer(num_words=len(d2v_model.wv.key_to_index) + 1, oov_token="<UNK>")
    tokenizer.fit_on_texts(df['Message'].values)
    
    X = tokenizer.texts_to_sequences(df['Message'].values)
    Y = pd.get_dummies(df['Category']).values
    X = pad_sequences(X, maxlen=50, padding='post', truncating='post')
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.15, random_state=42)
    return X_train, X_test, Y_train, Y_test, tokenizer, Y

X_train, X_test, Y_train, Y_test, tokenizer, Y = preprocess_data(df)

@st.cache_data
def create_embedding_matrix():
    embedding_matrix = np.zeros((len(d2v_model.wv.key_to_index) + 1, 20))
    for i in range(len(d2v_model.wv.key_to_index)):
        embedding_matrix[i] = d2v_model.wv[d2v_model.wv.index_to_key[i]]
    return embedding_matrix

embedding_matrix = create_embedding_matrix()

@st.cache_data
def train_model(X_train, Y_train, embedding_matrix):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(len(embedding_matrix), 20, weights=[embedding_matrix], trainable=True),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(2, activation="softmax")
    ])
    
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=20, batch_size=32, verbose=1)
    return model, history

model, history = train_model(X_train, Y_train, embedding_matrix)

@st.cache_data
def evaluate_model():
    loss, accuracy = model.evaluate(X_test, Y_test, verbose=0)
    yhat_classes = np.argmax(model.predict(X_test, verbose=0), axis=-1)
    rounded_labels = np.argmax(Y_test, axis=1)
    cm = confusion_matrix(rounded_labels, yhat_classes)
    return loss, accuracy, cm

user_input = st.text_area("📝 Enter your message here:")

if st.button("🔍 Classify"):
    if user_input:
        seq = tokenizer.texts_to_sequences([user_input])
        padded = pad_sequences(seq, maxlen=50, dtype='int32', value=0)
        pred = model.predict(padded)
        labels = ['ham', 'spam']
        st.write(f"📢 **The message is classified as:** **{labels[np.argmax(pred)]}**")
    else:
        st.warning("⚠️ Please enter a message.")

if st.checkbox("📊 Show Model Metrics"):
    loss, accuracy, cm = evaluate_model()

    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, linewidth=0.7, linecolor='cyan', fmt='g', ax=ax, cmap="BuPu")
    st.pyplot(fig)

    st.write(f"🔹 **Loss:** {loss:.4f}")
    st.write(f"🔹 **Accuracy:** {accuracy:.4f}")

    st.subheader("📈 Training History")
    
    fig_acc, ax_acc = plt.subplots()
    ax_acc.plot(history.history['accuracy'])
    ax_acc.set_title('Model Accuracy')
    ax_acc.set_ylabel('Accuracy')
    ax_acc.set_xlabel('Epoch')
    ax_acc.legend(['Train'], loc='upper left')
    st.pyplot(fig_acc)

    fig_loss, ax_loss = plt.subplots()
    ax_loss.plot(history.history['loss'])
    ax_loss.set_title('Model Loss')
    ax_loss.set_ylabel('Loss')
    ax_loss.set_xlabel('Epoch')
    ax_loss.legend(['Train'], loc='upper left')
    st.pyplot(fig_loss)
