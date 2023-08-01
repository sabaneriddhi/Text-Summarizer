import string
import math
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
# Tokenization function
def tokenize(text):
    return text.lower().split()

# Remove punctuation and stopwords
def preprocess_text(text):
    stop_words = set(stopwords.words('english') + list(string.punctuation))
    words = tokenize(text)
    return [word for word in words if word not in stop_words]

# Compute Term Frequency (TF) for each word in a sentence
def compute_tf(sentence, word_frequency):
    words = preprocess_text(sentence)
    sentence_length = len(words)
    tf = {word: words.count(word) / sentence_length for word in set(words)}
    return tf

# Compute Inverse Document Frequency (IDF) for each word in the document
def compute_idf(sentences, word_frequency):
    num_documents = len(sentences)
    idf = {}
    for sentence in sentences:
        words = preprocess_text(sentence)
        for word in set(words):
            idf[word] = idf.get(word, 0) + 1

    for word, frequency in idf.items():
        idf[word] = math.log(num_documents / (frequency + 1)) + 1

    return idf

# Compute TF-IDF scores for each word in a sentence
def compute_tfidf(sentence, tf, idf):
    tfidf = {word: tf[word] * idf[word] for word in tf}
    return tfidf

# Compute cosine similarity between two sentences
def cosine_similarity(sentence1, sentence2, tfidf1, tfidf2):
    numerator = 0
    for word in set(tfidf1.keys()) & set(tfidf2.keys()):
        numerator += tfidf1[word] * tfidf2[word]

    norm1 = math.sqrt(sum(tfidf1[word] ** 2 for word in tfidf1.keys()))
    norm2 = math.sqrt(sum(tfidf2[word] ** 2 for word in tfidf2.keys()))

    if norm1 != 0 and norm2 != 0:
        similarity = numerator / (norm1 * norm2)
    else:
        similarity = 0

    return similarity

# Extract the most important sentences based on cosine similarity scores
def extractive_summarizer(text, num_sentences=3):
    sentences = sent_tokenize(text)
    word_frequency = {}  # Keep track of word frequency for IDF computation

    # Preprocess and compute TF-IDF for each sentence
    sentence_tfidf = []
    for sentence in sentences:
        tf = compute_tf(sentence, word_frequency)
        idf = compute_idf(sentences, word_frequency)
        tfidf = compute_tfidf(sentence, tf, idf)
        sentence_tfidf.append(tfidf)

    # Compute cosine similarity for all sentence pairs
    similarity_matrix = []
    for i in range(len(sentences)):
        similarity_row = []
        for j in range(len(sentences)):
            similarity = cosine_similarity(
                sentences[i], sentences[j], sentence_tfidf[i], sentence_tfidf[j]
            )
            similarity_row.append(similarity)
        similarity_matrix.append(similarity_row)

    # Get the most important sentences using cosine similarity scores
    sentence_scores = [sum(similarity_matrix[i]) for i in range(len(similarity_matrix))]
    sentence_scores
    ranked_sentences = sorted(range(len(sentence_scores)), key=lambda x: sentence_scores[x], reverse=True)[:num_sentences]

    # Form the summary by combining the selected sentences
    summary = [sentences[idx] for idx in ranked_sentences]

    return ". ".join(summary)

input_text = """
Deep learning is part of a broader family of machine learning methods, which is based on artificial neural networks with representation learning. 
The adjective "deep" in deep learning refers to the use of multiple layers in the network. 
Methods used can be either supervised, semi-supervised or unsupervised.  
Deep-learning architectures such as deep neural networks, deep belief networks, deep reinforcement learning, recurrent neural networks, convolutional 
neural networks and transformers have been applied to fields including computer vision, speech recognition, natural language processing, machine translation, 
bioinformatics, drug design, medical image analysis, climate science, material inspection and board game programs, where they have produced results comparable 
to and in some cases surpassing human expert performance.Artificial neural networks (ANNs) were inspired by information processing and distributed communication 
nodes in biological systems. ANNs have various differences from biological brains. Specifically, artificial neural networks tend to be static and symbolic, 
while the biological brain of most living organisms is dynamic (plastic) and analog.Deep learning is a class of machine learning algorithms that
uses multiple layers to progressively extract higher-level features from the raw input. For example, in image processing, lower layers may identify edges, while higher 
layers may identify the concepts relevant to a human such as digits or letters or faces.From another angle to view deep learning, deep learning refers to
‘computer-simulate’ or ‘automate’ human learning processes from a source (e.g., an image of dogs) to a learned object (dogs). Therefore, a notion coined as 
“deeper” learning or “deepest” learning [9] makes sense. The deepest learning refers to the fully automatic learning from a source to a final learned object. 
A deeper learning thus refers to a mixed learning process: a human learning process from a source to a learned semi-object, followed by a computer learning process from 
the human learned semi-object to a final learned object.
"""

num_summary_sentences = 2
summary = extractive_summarizer(input_text, num_sentences=num_summary_sentences)
print(summary)








