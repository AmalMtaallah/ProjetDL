from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import torchvision.transforms as transforms
import io

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle PyTorch
model_path = "model/model.pth"
model = torch.load(model_path, map_location=torch.device('cpu'))
model.eval()

# Préparer les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Classes prédéfinies
classes = ["Benign", "Malignant"]

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Lire l'image depuis la requête
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Appliquer les transformations
        input_tensor = transform(image).unsqueeze(0)

        # Faire une prédiction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, predicted = torch.max(outputs, 1)

        # Obtenir le résultat
        result = classes[predicted.item()]

        # Réponse JSON avec "success" et "message"
        return JSONResponse(content={"success": 1, "message": f"Prediction: {result}"})

    except Exception as e:
        # En cas d'erreur
        return JSONResponse(content={"success": 0, "message": str(e)}, status_code=500)
