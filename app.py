import io
import torch
from torchvision import models, transforms
from PIL import Image
from flask import Flask, request, jsonify

app = Flask(__name__)

model = models.resnet18(pretrained=True)
model.eval()


with open("imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  
        std=[0.229, 0.224, 0.225]    
    )
])

@app.route('/predict', methods=['POST'])
def predict():
    if 'InputImage' not in request.files:
        return jsonify({
            "Status": [{"MessageCode": "E", "MessageText": "No InputImage file part"}],
            "ReturnData": [],
            "DevelopedBy": "Vinod Kumar"
        }), 400

    file = request.files['InputImage']
    if file.filename == '':
        return jsonify({
            "Status": [{"MessageCode": "E", "MessageText": "No selected file"}],
            "ReturnData": [],
            "DevelopedBy": "Vinod Kumar"
        }), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return jsonify({
            "Status": [{"MessageCode": "E", "MessageText": "Invalid image file"}],
            "ReturnData": [],
            "DevelopedBy": "Vinod Kumar"
        }), 400

    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)

    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    top_prob, top_catid = torch.topk(probabilities, 1)
    pred_class = classes[top_catid.item()]

    return jsonify({
        "Status": [{"MessageCode": "S", "MessageText": "OK"}],
        "ReturnData": [{"PredectedObject": pred_class}],
        "DevelopedBy": "Vinod Kumar"
    })

if __name__ == '__main__':
    app.run(debug=True)
