import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from transformers import pipeline

# 1. Check device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"=== Running on device: {device.upper()} ===\n")

# 2. Initialize Preprocessing (MTCNN) & Verification Model (InceptionResnetV1)
mtcnn = MTCNN(image_size=160, margin=20, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# 3. Initialize Deepfake Detection Model (ViT)
detector = pipeline(
    "image-classification",
    model="dima806/deepfake_vs_real_image_detection",
    device=0 if device == 'cuda' else -1
)

# 4. Load two local face images from your project folder
img1 = Image.open("face1.jpg").convert("RGB")
img2 = Image.open("face2.jpg").convert("RGB")

# 5. Run Verification Inference Test
face1 = mtcnn(img1)
face2 = mtcnn(img2)

if face1 is not None and face2 is not None:
    emb1 = resnet(face1.unsqueeze(0).to(device))
    emb2 = resnet(face2.unsqueeze(0).to(device))
    cos_sim = torch.nn.functional.cosine_similarity(emb1, emb2).item()
    print(f"[Verification Model Test] FaceNet Cosine Similarity: {cos_sim:.4f}")
else:
    print("[Verification Model Test] Face not detected in one of the images.")

# 6. Run Deepfake Detection Inference Test
print("\n[Deepfake Detection Test - Image 1]:")
for res in detector(img1):
    print(f"  Label: {res['label']} | Confidence: {res['score']:.4f}")

print("\n[Deepfake Detection Test - Image 2]:")
for res in detector(img2):
    print(f"  Label: {res['label']} | Confidence: {res['score']:.4f}")
