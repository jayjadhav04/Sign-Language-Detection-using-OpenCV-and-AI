# Sign-Language-Detection-using-OpenCV-and-AI

This project enables real-time sign language detection using a webcam, OpenCV, and a trained AI model.



********** Unique Feature **********

Unlike other sign language detection projects, this system dynamically creates folders based on user input. When collecting data, the system prompts the user to enter the sign name. It then automatically creates a folder for that sign and saves all captured images in the respective folder. This makes data organization more efficient and customizable for training.




********** Features **********

✅ Collects hand sign images for training

✅ Uses AI to classify hand gestures in real time

✅ Displays recognized signs on the screen

✅ Supports multiple sign categories




********** How to Use **********

Step 1: Collect Data

Run the script to collect hand sign images for training:
python datacollection.py
Enter the name of the sign when prompted.
Press 's' to save an image.    Press 'q' to exit.

Step 2: Train the Model

Train a deep learning model (e.g., using TensorFlow or Teachable Machine) and save it as keras_model.h5 with labels.txt.

Step 3: Run the Detection System

Start the real-time sign language detection:
python test.py
The detected sign will be displayed on the screen.
Press 'q' to exit.



********** Contributing **********

Feel free to fork the repo and improve the project! 🚀
