#Dogs and Cats Classifier 

A project for my Deep Learning module, developed by a 2nd-year student. This **model** classifies images of dogs and cats with ~98% accuracy.

#Live Demo

You can test my model online! Upload your picture in an interactive web-based app, hosted on *Hugging Face Spaces*.

*Demo: (https://huggingface.co/spaces/christianjordan/DogsAndCatClass)*

--------

#Technologies Used

* *Python*
* *TensorFlow & Keras:* For building and **training** the *model*.
* *Transfer Learning:* Using the MobileNetV2 architecture as a feature extractor.
* *Pandas & Matplotlib:* For **analyzing** and visualizing results.
* *Gradio:* For creating the interactive web-based interface.
* *Google Colab:* For training with a T4 GPU.
* *Hugging Face Spaces:* For deployment of the web application.
* *Git & Git LFS:* For version control and managing large model files.

---------

# Results

The *model* was *trained* for 20 epochs, achieving ~98.5% accuracy on the validation set.

<img width="689" height="701" alt="download" src="https://github.com/user-attachments/assets/b2ccca21-066d-4de7-8992-2480d631c56e" />


---

# Project Structure

* 'Dogs_vs_Cats_Training.ipynb': The Jupyter Notebook with the full code for data loading, preprocessing, **augmentation**, and model *training*.
* 'dogs_vs_cats_classifier.keras': The *trained model* (9.2MB), tracked with Git LFS.
* 'app.py': The code for the Gradio web application.
* 'requirements.txt': The Python libraries required to run 'app.py'.
