import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms

# model = models.googlenet(pretrained=True)
# model.eval()
from model import FrameLevelTransformer

google_net = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "googlenet",
    weights=models.GoogLeNet_Weights.IMAGENET1K_V1,
)
# Remove the last layers to get pool5 features
google_net = torch.nn.Sequential(*list(google_net.children())[:-2])
google_net.eval()

transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def preprocess(input_image):
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = transform(input_image)
    return input_image


def get_visual_features(video_path: str, frame_rate: int = 2):
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % frame_rate == 0:
            frame = preprocess(frame)
            frames.append(frame)
    cap.release()

    frames = torch.stack(frames)
    with torch.no_grad():
        features = google_net(frames)

    # Average pool features along time axis
    features = torch.mean(features, dim=2)

    return features.numpy()


if __name__ == "__main__":
    video_path = "../tvsum50_ver_1_1/ydata-tvsum50-v1_1/video/sTEELN-vY30.mp4"
    frame_rate = 2
    visual_features = get_visual_features(video_path, frame_rate)
    print(visual_features)
