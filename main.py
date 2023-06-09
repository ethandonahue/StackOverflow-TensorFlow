import numpy as np
import tensorflow as tf

# Load the saved model
loaded_model = tf.keras.models.load_model("text_categorization_model")

while True:
    question = input("Enter your question: ")

    # Create an array with the user's question
    question = [question]

    category_labels = ['csharp', 'java', 'javascript', 'python']
    prediction = loaded_model.predict(question)
    print(prediction)
    category = category_labels[np.argmax(prediction)]
    print(category)
