🧠 CAPTCHA Solver API using FastAPI

This is my first project using FastAPI, built with Python 3 and fully containerized with Docker. It leverages machine learning and computer vision techniques to train a model capable of breaking simple CAPTCHA images.

The project exposes an API that receives CAPTCHA images (Base64 format), processes them, and returns the extracted text using a pre-trained model.
🔧 Tech Stack

    Python 3
    FastAPI for API development
    Docker for environment isolation and deployment

    Machine Learning & Image Processing Libraries:
        scikit-learn
        scikit-image
        matplotlib
        opencv-python-headless
        numpy

📌 Features

    Train a model to recognize characters from CAPTCHA images using SVM (Support Vector Machine).
    Expose an API endpoint to receive a Base64-encoded image and return the predicted text.
    Preprocess images to enhance accuracy (noise reduction, binarization, etc.).
    Designed for learning and experimenting with ML + API integration.

🚀 Getting Started
You can run the project using Docker:
`docker compose up --build`

Once running, access the API docs at:
`http://localhost:8000/docs`

You can use this url for check mongoDB is connectable:
`http://localhost:8000/ping-db`