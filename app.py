
import gradio as gr
import numpy as np
import tensorflow as tf

try:
    model = tf.keras.models.load_model('dogs_vs_cats_classifier.keras')
except Exception as e:
    print(f"Error to loading: {e}")
    raise e

IMAGE_SIZE = (160, 160)
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def classify_image(input_image):
    if input_image is None:
        return {"Грешка": 1.0}
    img_resized = tf.image.resize(input_image, IMAGE_SIZE)
    img_batch = np.expand_dims(img_resized, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    prediction_logits = model.predict(img_preprocessed)
    score = tf.nn.sigmoid(prediction_logits[0][0]).numpy()
    results = {
        "Cat 🐱": float(1 - score),
        "Dog 🐶": float(score)
    }
    return results
iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="numpy"),
    outputs=gr.Label(num_top_classes=2),
    title="Classifier for Dogs and Cats",
    description="Upload a picture and check what module thing is it - Dog or Cat. Project from a 2nd year Computer Science student."
)
iface.launch()
