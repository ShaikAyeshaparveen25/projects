#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np


# In[1]:


import tkinter as tk
from tkinter import messagebox  # Import messagebox
import cv2
import os  # Import os for directory creation
from PIL import Image  # Import Image for image processing
import numpy as np  # Import numpy for array operations
import mysql.connector  # Import MySQL connector

# Initialize the main window
window = tk.Tk()
window.title("Face Recognition System")

# Create labels and entry fields for user input
ll = tk.Label(window, text="Name", font=("Algerian", 20))
ll.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

def train_classifier():
    data_dir = "C:/Users/shaik/Downloads/face recognition/data"

    def train_classifier_inner(data_dir, output_file='classifier.xml'):
        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        faces = []
        ids = []
        
        for image in path:
            try:
                img = Image.open(image).convert('L')
                imageNp = np.array(img, 'uint8')
                id = int(os.path.split(image)[1].split(".")[1])
                faces.append(imageNp)
                ids.append(id)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
        
        if len(faces) == 0:
            print("No faces found. Make sure your data directory contains valid images.")
            return
        
        ids = np.array(ids)
        
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)

        clf.write(output_file)
        messagebox.showinfo('Result', 'Training dataset completed')

    train_classifier_inner(data_dir)

# Button to trigger training
b1 = tk.Button(window, text="Training", font=("Algerian", 20), bg='orange', fg='red', command=train_classifier)
b1.grid(column=0, row=4)

def detect_face():
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        
        coords = []
        
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = 100 * (1 - pred / 300)

            if confidence > 75:
                if id == 1:
                    name = "zaheer"
                elif id == 2:
                    name = "Jabeen"
                elif id == 3:
                    name = "Afreen"
                else:
                    name = "UNKNOWN"
            else:
                name = "UNKNOWN"

            cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords = [x, y, w, h]

        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    video_capture = cv2.VideoCapture(0)
    while True:
        ret, img = video_capture.read()
        if not ret:
            break

        img = recognize(img, clf, faceCascade)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:  # Enter key to exit
            break

    video_capture.release()
    cv2.destroyAllWindows()

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
    else:
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="afreen",
            database="Authorized_user"
        )
        mycursor = mydb.cursor()
        mycursor.execute("SELECT * FROM my_table")
        myresult = mycursor.fetchall()
        id = 1
        for x in myresult:
            id += 1
        sql = "INSERT INTO my_table(id, Name, Age, Address) VALUES (%s, %s, %s, %s)"
        val = (id, t1.get(), t2.get(), t3.get())
        mycursor.execute(sql, val)
        mydb.commit()

        face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

        def face_cropped(img):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)

            if len(faces) == 0:
                return None

            for (x, y, w, h) in faces:
                cropped_face = img[y:y+h, x:x+w]
                return cropped_face

        cap = cv2.VideoCapture(0)
        img_id = 3

        if not os.path.exists("data"):
            os.makedirs("data")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break

            cropped_face = face_cropped(frame)
            if cropped_face is not None:
                img_id += 1
                face = cv2.resize(cropped_face, (200, 200))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                file_name_path = f"data/user.{id}.{img_id}.jpg"
                cv2.imwrite(file_name_path, face)
                cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow("Cropped face", face)

                if img_id >= 200:
                    break

            if cv2.waitKey(1) == 13:  # Stop if 'Enter' key is pressed
                break

        cap.release()
        cv2.destroyAllWindows()
        messagebox.showinfo('Result', 'Generating dataset completed!!')

# Button to trigger face detection
b2 = tk.Button(window, text="Detect the face", font=("Algerian", 20), bg='green', fg='white', command=detect_face)
b2.grid(column=1, row=4)

# Button to generate dataset
b3 = tk.Button(window, text="Generate dataset", font=("Algerian", 20), bg='pink', fg='black', command=generate_dataset)
b3.grid(column=2, row=4)

# Set the size of the window
window.geometry("800x200")
window.mainloop()


# In[4]:


import tkinter as tk
from tkinter import messagebox
import cv2
import os
from PIL import Image
import numpy as np
import mysql.connector

window = tk.Tk()
window.title("Face Recognition System")

# Labels and Entry fields
ll = tk.Label(window, text="Name", font=("Algerian", 20))
ll.grid(column=0, row=0)
t1 = tk.Entry(window, width=50, bd=5)
t1.grid(column=1, row=0)

l2 = tk.Label(window, text="Age", font=("Algerian", 20))
l2.grid(column=0, row=1)
t2 = tk.Entry(window, width=50, bd=5)
t2.grid(column=1, row=1)

l3 = tk.Label(window, text="Address", font=("Algerian", 20))
l3.grid(column=0, row=2)
t3 = tk.Entry(window, width=50, bd=5)
t3.grid(column=1, row=2)

def train_classifier():
    data_dir = "C:/Users/shaik/Downloads/face recognition/data"

    def train_classifier_inner(data_dir, output_file='classifier.xml'):
        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        faces = []
        ids = []
        
        for image in path:
            try:
                img = Image.open(image).convert('L')
                img = cv2.equalizeHist(np.array(img))  # Histogram equalization
                imageNp = np.array(img, 'uint8')
                id = int(os.path.split(image)[1].split(".")[1])
                faces.append(imageNp)
                ids.append(id)
            except Exception as e:
                print(f"Error processing image {image}: {e}")
        
        if len(faces) == 0:
            print("No faces found. Make sure your data directory contains valid images.")
            return
        
        ids = np.array(ids)
        
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)

        clf.write(output_file)
        messagebox.showinfo('Result', 'Training dataset completed')

    train_classifier_inner(data_dir)

b1 = tk.Button(window, text="Training", font=("Algerian", 20), bg='orange', fg='red', command=train_classifier)
b1.grid(column=0, row=4)

is_detecting = False
stop_button = None

def detect_face():
    global is_detecting, stop_button
    is_detecting = True

    # Create Stop Detection button
    stop_button = tk.Button(window, text="Stop Detection", font=("Algerian", 20), bg='red', fg='white', command=stop_detection)
    stop_button.grid(column=3, row=4)

    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("classifier.xml")

    def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
        gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)
        
        coords = []
        
        for (x, y, w, h) in features:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            id, pred = clf.predict(gray_image[y:y+h, x:x+w])
            confidence = 100 * (1 - pred / 300)

            mydb = mysql.connector.connect(
                host="localhost",
                user="root",
                port=4306,
                passwd="",
                database="Authorized_user"
            )
            mycursor = mydb.cursor()
            mycursor.execute("SELECT name FROM my_table WHERE id=%s", (id,))
            s = mycursor.fetchone()
            mycursor.close()
            mydb.close()

            name = "UNKNOWN"
            if s is not None:
                name = s[0]

            if confidence > 70:  # Adjust confidence threshold for better accuracy
                cv2.putText(img, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            else:
                cv2.putText(img, "UNKNOWN", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 1, cv2.LINE_AA)
            coords = [x, y, w, h]

        return coords

    def recognize(img, clf, faceCascade):
        coords = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
        return img

    video_capture = cv2.VideoCapture(0)
    while is_detecting:
        ret, img = video_capture.read()
        if not ret:
            break

        img = recognize(img, clf, faceCascade)
        cv2.imshow("Face Detection", img)

        if cv2.waitKey(1) == 13:  # Enter key to exit
            break

    video_capture.release()
    cv2.destroyAllWindows()
    stop_button.grid_forget()  # Hide the Stop Detection button
    is_detecting = False  # Reset flag when done

def stop_detection():
    global is_detecting
    is_detecting = False
    cv2.destroyAllWindows()  # Ensure any open windows are closed

def generate_dataset():
    if t1.get() == "" or t2.get() == "" or t3.get() == "":
        messagebox.showinfo('Result', 'Please provide complete details of the user')
        return
    
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        port=4306,
        passwd="",
        database="Authorized_user"
    )
    
    mycursor = mydb.cursor()

    # Get the next available ID
    mycursor.execute("SELECT * FROM my_table")
    myresult = mycursor.fetchall()
    id = len(myresult) + 1  # Use length of results to determine ID
    sql = "INSERT INTO my_table(id, Name, Age, Address) VALUES (%s, %s, %s, %s)"
    val = (id, t1.get(), t2.get(), t3.get())  # Corrected to t3.get()
    mycursor.execute(sql, val)
    mydb.commit()
    
    face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    def face_cropped(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return None
        for (x, y, w, h) in faces:
            cropped_face = img[y:y+h, x:x+w]
            return cropped_face

    cap = cv2.VideoCapture(0)
    img_id = 0  # Start with zero to increment

    if not os.path.exists("data"):
        os.makedirs("data")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        cropped_face = face_cropped(frame)
        if cropped_face is not None:
            img_id += 1
            face = cv2.resize(cropped_face, (200, 200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path = f"data/user.{id}.{img_id}.jpg"
            cv2.imwrite(file_name_path, face)
            cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Cropped face", face)

            if img_id >= 200:  # Limit to 200 images
                break

        if cv2.waitKey(1) == 13:  # Stop if 'Enter' key is pressed
            break

    cap.release()
    cv2.destroyAllWindows()
    
    messagebox.showinfo('Result', 'Generating dataset completed!!')

    mycursor.close()  # Close cursor
    mydb.close()      # Close database connection

b2 = tk.Button(window, text="Detect the face", font=("Algerian", 20), bg='green', fg='white', command=detect_face)
b2.grid(column=1, row=4)

b3 = tk.Button(window, text="Generate dataset", font=("Algerian", 20), bg='pink', fg='black', command=generate_dataset)
b3.grid(column=2, row=4)

window.geometry("800x200")
window.mainloop()


# In[14]:





# In[ ]:




