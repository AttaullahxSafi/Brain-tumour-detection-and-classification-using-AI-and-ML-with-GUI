from tkinter import *
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import tensorflow as tf
from keras.models import load_model
import webbrowser

COPYRIGHT = """Copyright © 2024 All rights reserved 
By: Attaullah Shahbazkail"""

TITLE = """Brain Tumour Detection and Classification
 (through AI and ML) System"""

BCU_image_path = "bcu_logo.jpg"

BCU_IMAGE = Image.open(BCU_image_path)

BRAIN_TUMOR_INFORMATION_URL ="https://www.cancerresearchuk.org/about-cancer/brain-tumours"

HEALTH_INFORMATION_URL_DICT = {
    "Glioma Tumor": "https://www.cancerresearchuk.org/about-cancer/brain-tumours/types/glioma-adults",
    "Meningioma Tumor": "https://www.cancerresearchuk.org/about-cancer/brain-tumours/types/meningioma",
    "No Tumor": BRAIN_TUMOR_INFORMATION_URL,
    "Pituitary Tumor": "https://www.cancerresearchuk.org/about-cancer/brain-tumours/types/pituitary-tumours"
}
# Initialise class names
class_names = ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor', 'Pituitary Tumor']

# Load brain tumor detection model
cnn_model = load_model('final_cnn_model.h5')

# pre-preprocessing image function
def load_and_preprocess(filename, img_shape=224):
    img = tf.io.read_file(filename)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, size=[img_shape, img_shape])
    img = img / 255.
    return img


# Prediction function
def make_prediction(model, classes, filename):
    img = load_and_preprocess(filename)
    pred = model.predict(tf.expand_dims(img, axis=0))
    pred = np.array(pred)
    pred = np.argmax(pred)
    pred = int(pred)
    pred_class = classes[pred]
    return pred_class


# display results function
def display_results(class_predicted):
    if class_predicted != "No Tumor":
        result_label.config(text=f"{class_predicted}", bg="#FAE100")
    else:
        result_label.config(text=f"{class_predicted}", bg="#78BE20")

    text1.config(text="Successfully Uploaded. Please to see the results on the right panel.")

    if class_predicted == "No Tumor":
        health_information_label.config(text="For Health Information")
    elif class_predicted != "No Tumor" and class_predicted == 'Glioma Tumor':
        health_information_label.config(text="For Glioma Tumor Information")
    elif class_predicted != "No Tumor" and class_predicted == 'Meningioma Tumor':
        health_information_label.config(text="For Meningioma Tumor Information")
    elif class_predicted != "No Tumor" and class_predicted == 'Pituitary Tumor':
        health_information_label.config(text="For Pituitary Tumor Information")

def upload_and_display_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        opened_img = Image.open(file_path)
        img = opened_img.resize((562, 470))
        img = ImageTk.PhotoImage(img)
        mri_label.config(image=img)
        mri_label.image = img
        predicted_class = make_prediction(cnn_model, class_names, file_path)
        display_results(predicted_class)

def open_website_link():
    predicted_class = result_label.cget("text")
    if predicted_class in HEALTH_INFORMATION_URL_DICT:
        webbrowser.open(HEALTH_INFORMATION_URL_DICT[predicted_class])
    else:
        webbrowser.open(BRAIN_TUMOR_INFORMATION_URL)

window = tk.Tk()
window.title("Brain Tumor Detection and Classification System")
window.config(background="white")
window.minsize(width=1000, height=725)
window.resizable(width=False, height=False)
window.iconbitmap('brain.ico')

program_title = tk.Label(text=TITLE, font=("Times New Roman", 18, "bold"), background="white")
program_title.place(x=425, y=10)

text1 = tk.Label(text="Please upload an image of the MRI scan using the button below:", font=("Times New Roman", 16),
                 background="#FDFD96", width=47, height=2, padx=2, anchor="w")
text1.place(x=20, y=108)

text2 = tk.Label(text="MRI scan result", font=("Times New Roman", 16, "bold"), background="#005EB8",
                 foreground="white", width=30, height=2, padx=5, anchor="w", highlightbackground="black",
                 highlightthickness=2)
text2.place(x=600, y=108)

result_label = tk.Label(window, text="Result will be shown here", font=("Times New Roman", 16),
                        background="#CCE3F5", width=30, height=2,
                        padx=5, anchor="w", highlightbackground="black", highlightthickness=2)
result_label.place(x=600, y=158)

accuracy_label = tk.Label(text="Model's Accuracy", font=("Times New Roman", 16, "bold"), background="#005EB8",
                          foreground="white", width=30, height=2, anchor="w", padx=5,
                          highlightbackground="black", highlightthickness=2)
accuracy_label.place(x=600, y=218)


health_information_label = tk.Label(text="For Health Information", font=("Times New Roman", 16, "bold"),
                          background="#005EB8", foreground="white", width=30, height=2, anchor="w", padx=5,
                          highlightbackground="black", highlightthickness=2)
health_information_label.place(x=600, y=330)

information_button = tk.Button(window, text="Click Here", command=open_website_link,
                          font=("Times New Roman", 14, "bold"), background="#CCE3F5", highlightthickness=0, border=0,
                          fg="black", width=33, padx=7, pady=4)
information_button.place(x=600, y=388)

copyright_label = tk.Label(text=COPYRIGHT, font=("Times New Roman", 14, "bold"), background="white",
                           foreground="black", width=30, height=2)
copyright_label.place(x=630, y=630)

percentage_label = tk.Label(window, text="98%", font=("Times New Roman", 16), background="#CCE3F5", width=30, height=2,
                            padx=5, anchor="w", highlightbackground="black", highlightthickness=2)
percentage_label.place(x=600, y=268)

mri_placeholder_label = tk.Label(window, text="UPLOAD MRI IMAGE", foreground="white",
                                 background="grey", width=80, height=31)
mri_placeholder_label.place(x=25, y=165)

mri_label = tk.Label(window, background="grey")
mri_label.place(x=25, y=165)

BCU_IMAGE = BCU_IMAGE.resize(size=(285, 80))
tk_image = ImageTk.PhotoImage(BCU_IMAGE)

bcu_label = tk.Label(window, image=tk_image, border=0)
bcu_label.place(x=25, y=5)

upload_button = tk.Button(window, text="Upload", command=upload_and_display_image,
                          font=("Times New Roman", 15), background="#005EB8", fg="white", border=0, padx=5, pady=5)
upload_button.place(x=255, y=650)
window.mainloop()
