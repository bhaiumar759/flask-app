Vessel Mask Generator
A Flask application to upload grayscale medical images, process them to generate vessel masks using OpenCV, and display the results.
Project Structure
vessel_mask_project/
├── app.py              # Flask application
├── requirements.txt    # Python dependencies
├── static/
│   └── uploads/       # Directory for uploaded and processed images
├── templates/
│   └── index.html     # HTML template for the web interface
└── README.md

Prerequisites

Python 3.8+

Setup Instructions

Navigate to the project directory:cd vessel_mask_project


Create a virtual environment and activate it:python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate


Install dependencies:pip install -r requirements.txt


Run the Flask server:python app.py



Usage

Open your browser and navigate to http://localhost:5000.
Upload a grayscale medical image (JPEG or PNG).
Click "Upload and Process" to generate the vessel mask.
View the original image, intermediate morphology result, and final vessel mask.

Notes

The application saves uploaded and processed images in static/uploads.
Ensure the uploaded image is a valid grayscale medical image (e.g., angiogram) for best results.
Uses Tailwind CSS (via CDN) for styling.

