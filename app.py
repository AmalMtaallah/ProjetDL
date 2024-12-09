from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io

# Initialiser l'application FastAPI
app = FastAPI()

# Charger le modèle PyTorch
model_path = "model/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(model_path, map_location=device)
model = model.to(device)
model.eval()

# Préparer les transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Redimensionner
    transforms.ToTensor(),         # Convertir en tenseur
    transforms.Normalize(          # Normalisation selon les statistiques ImageNet
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

@app.post("/predict/")
async def predict_image(file: UploadFile = File(...)):
    try:
        # Lire l'image depuis la requête
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Appliquer les transformations
        input_tensor = transform(image).unsqueeze(0).to(device)

        # Faire une prédiction
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.sigmoid(outputs).squeeze().item()  # Appliquer sigmoïde pour obtenir les probabilités

        # Décoder le résultat
        prediction = "Malignant" if probs >= 0.5 else "Benign"
        confidence = probs if prediction == "Malignant" else 1 - probs

        # Réponse JSON avec "success" et "message"
        return JSONResponse(content={
            "success": 1,
            "message": f"Prediction: {prediction}",
            "confidence": f"{confidence:.2f}"
        })

    except Exception as e:
        # En cas d'erreur
        return JSONResponse(content={"success": 0, "message": str(e)}, status_code=500)
