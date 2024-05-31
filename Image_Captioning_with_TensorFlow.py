import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, TimeDistributed, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

# 1. Data Loading and Preprocessing

def load_data(filename):
    """Loads image filenames and captions from a file."""
    with open(filename, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        image_paths = [line.strip().split(',')[0] for line in lines]
        captions = [line.strip().split(',')[1:] for line in lines]
    return image_paths, captions

# Load the Flickr8k dataset
image_paths, captions = load_data('flickr8k.txt')  # Replace with your dataset file

# Define maximum caption length
max_len = 50

# Tokenize captions
tokenizer = Tokenizer(num_words=5000)  # Keep top 5000 words
tokenizer.fit_on_texts([cap for sublist in captions for cap in sublist])
vocab_size = len(tokenizer.word_index) + 1

# Function to convert captions to sequences
def captions_to_sequences(captions):
    """Converts captions to sequences of integers."""
    sequences = []
    for caption_list in captions:
        for caption in caption_list:
            sequences.append(tokenizer.texts_to_sequences([caption])[0])
    return sequences

# Convert captions to sequences
captions_sequences = captions_to_sequences(captions)
padded_captions = pad_sequences(captions_sequences, maxlen=max_len, padding='post')

# 2. Model Building (Encoder-Decoder)

# Define encoder (ResNet50)
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
inputs = Input(shape=(224, 224, 3))
encoder_outputs = resnet_model(inputs)
encoder_outputs = Flatten()(encoder_outputs)
encoder = Model(inputs, encoder_outputs)

# Define decoder (RNN with LSTM)
decoder_inputs = Input(shape=(max_len-1,))
decoder_embedding = Embedding(vocab_size, 256)(decoder_inputs)
decoder_lstm = LSTM(256, return_sequences=True)(decoder_embedding)
decoder_dense = TimeDistributed(Dense(vocab_size, activation='softmax'))(decoder_lstm)
decoder = Model(decoder_inputs, decoder_dense)

# Combine encoder and decoder into a single model
decoder_outputs = decoder(encoder(inputs))
model = Model(inputs, decoder_outputs)

# 3. Training

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Image data loading and preprocessing (Placeholder - You'll need to adapt this)
def load_image(image_path):
    """Loads and preprocesses an image."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Create training data
train_images = [load_image(os.path.join('images', path)) for path in image_paths]
train_images = np.array(train_images)

# One-hot encode captions for training
train_captions = to_categorical(padded_captions, num_classes=vocab_size)
train_captions = train_captions[:, :-1]  # Remove last word for prediction

# Train the model
model.fit(train_images, train_captions, epochs=10, batch_size=32)

# 4. Evaluation

def generate_caption(image_path, model):
    """Generates a caption for an image."""
    image = load_image(image_path)
    image = np.expand_dims(image, axis=0)
    
    # Encode the image
    image_features = encoder.predict(image)
    
    # Start with the start token
    caption_sequence = [tokenizer.word_index['start']]
    
    # Predict words one by one until the end token
    for _ in range(max_len-1):
        predicted_probs = model.predict([image_features])
        predicted_word_index = np.argmax(predicted_probs[0, -1, :])
        
        caption_sequence.append(predicted_word_index)
        
        # If end token is predicted, stop
        if predicted_word_index == tokenizer.word_index['end']:
            break
    
    # Decode the predicted caption
    predicted_caption = tokenizer.sequences_to_texts([caption_sequence])[0]
    return predicted_caption

# Example:
image_path = 'path/to/image.jpg'
predicted_caption = generate_caption(image_path, model)
print(f"Predicted Caption: {predicted_caption}")

# 5. Visualization (Optional)

def visualize_predictions(image_paths, model, num_to_show=5):
    """Visualizes predicted captions for a few images."""
    plt.figure(figsize=(15, 10))
    for i in range(num_to_show):
        image_path = image_paths[i]
        predicted_caption = generate_caption(image_path, model)
        
        plt.subplot(num_to_show, 1, i+1)
        img = Image.open(image_path)
        plt.imshow(img)
        plt.title(f"Predicted Caption: {predicted_caption}")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Example:
visualize_predictions(image_paths, model)