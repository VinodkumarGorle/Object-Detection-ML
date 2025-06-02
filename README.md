
# 🖼️ Image Classification API using Flask

This is a simple Flask-based REST API for image classification using a pretrained ResNet18 model from PyTorch. The API accepts an input image and returns the top predicted object class based on the ImageNet dataset.

## 📁 Project Files

```
.
├── app.py                 # Flask app with prediction endpoint
├── imagenet_classes.txt  # List of ImageNet class labels (1000 classes)
├── README.md              # Project documentation
```

## ✅ Requirements

- Python 3.7+
- pip

## 📦 Installation

### 1. Clone the Repository

```bash
[git clone https://github.com/yourusername/image-classification-api.git](https://github.com/VinodkumarGorle/Object-Detection-ML-API.git)
cd Object-Detection-ML-API
```

### 2. Create a Virtual Environment (Optional)

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision flask pillow
```

### 4. Ensure `imagenet_classes.txt` Exists

This file should contain 1000 class labels from ImageNet. If it's missing, you can generate it using:

```python
# Run this script to create the file
from torchvision import models

labels = []
with open("imagenet_classes.txt", "w") as f:
    # Download from official source or manually add labels
    # Alternatively, use torchvision helper if available
    # Example below assumes you have a list of labels
    for label in labels:
        f.write(f"{label}\n")
```

Or download it from: https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt

## 🚀 Running the App

```bash
python app.py
```

Server will start at: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## 📡 API Endpoint

### POST `/predict`

#### Request

- **Content-Type**: `multipart/form-data`
- **Form field**: `InputImage` (image file)

#### Example with `curl`

```bash
curl -X POST -F "InputImage=@dog.jpg" http://127.0.0.1:5000/predict
```

#### Success Response

```json
{
  "Status": [{"MessageCode": "S", "MessageText": "OK"}],
  "ReturnData": [{"PredectedObject": "Labrador retriever"}],
  "DevelopedBy": "Vinod Kumar"
}
```

#### Error Responses

- **Missing file field**
```json
{
  "Status": [{"MessageCode": "E", "MessageText": "No InputImage file part"}],
  "ReturnData": [],
  "DevelopedBy": "Vinod Kumar"
}
```

- **Invalid image file**
```json
{
  "Status": [{"MessageCode": "E", "MessageText": "Invalid image file"}],
  "ReturnData": [],
  "DevelopedBy": "Vinod Kumar"
}
```

## 👨‍💻 Developer

- **Name**: Vinod Kumar
- **Tech Used**: Python, Flask, PyTorch, Torchvision, PIL

## 📌 Notes

- Make sure `imagenet_classes.txt` is aligned with the ResNet18 model output.
- This project is intended for educational/demonstration purposes.
- For production, consider using Gunicorn, Docker, and adding logging/error-handling.

