import json

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, ToTensor

import config
import features.visual
from features import get_audio_embeddings
from model import VideoSummarizationModel


class CustomDataset(Dataset):
    def __init__(
        self, video_list, visual_extraction_func, audio_extraction_func, transform=None
    ):
        self.video_list = video_list
        self.visual_extraction_func = visual_extraction_func
        self.audio_extraction_func = audio_extraction_func
        self.transform = transform

    def __len__(self):
        return len(self.video_list)

    def __getitem__(self, idx):
        video_name = self.video_list[idx]

        visual_features = self.visual_extraction_func(video_name)
        audio_features = self.audio_extraction_func(video_name)

        # Apply transformations if specified
        if self.transform is not None:
            visual_features = self.transform(visual_features)
            audio_features = self.transform(audio_features)

        return visual_features, audio_features


batch_size = config.DATALOADER_BATCH_SIZE
learning_rate = config.LEARNING_RATE
num_epochs = config.EPOCHS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open(config.TVSUM_SPLIT, "r") as f:
    tvsum_splits = json.load(f)

with open(config.SUMME_SPLIT, "r") as f:
    summe_splits = json.load(f)

# There are 5 random splits
train_split = tvsum_splits[0]["train"]
test_split = tvsum_splits[0]["test"]

train_dataset = CustomDataset(
    train_split,
    transform=Compose([ToTensor()]),
    visual_extraction_func=features.get_visual_features,
    audio_extraction_func=get_audio_embeddings,
)
test_dataset = CustomDataset(
    test_split,
    transform=Compose([ToTensor()]),
    visual_extraction_func=features.get_visual_features,
    audio_extraction_func=get_audio_embeddings,
)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

visual_input_dim = config.VISUAL_INPUT_DIM
hidden_dim = config.HIDDEN_DIM
num_heads = config.N_HEADS  # Adjust based on the desired number of attention heads
model = VideoSummarizationModel(visual_input_dim, hidden_dim, num_heads)
model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

best_loss = float("inf")
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0

    for batch_idx, (
        visual_features,
        audio_features,
        shots_boundaries,
        targets,
    ) in enumerate(train_dataloader):
        visual_features = visual_features.to(device)
        audio_features = audio_features.to(device)
        shots_boundaries = shots_boundaries.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        outputs = model(visual_features, audio_features, shots_boundaries)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    average_loss = total_loss / len(train_dataloader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}")

    if average_loss < best_loss:
        best_loss = average_loss
        torch.save(model.state_dict(), "m.pt")

model.eval()
total_test_loss = 0.0

with torch.no_grad():
    for batch_idx, (
        visual_features,
        audio_features,
        shots_boundaries,
        targets,
    ) in enumerate(test_dataloader):
        visual_features = visual_features.to(device)
        audio_features = audio_features.to(device)
        shots_boundaries = shots_boundaries.to(device)
        targets = targets.to(device)

        outputs = model(visual_features, audio_features, shots_boundaries)

        loss = criterion(outputs, targets)
        total_test_loss += loss.item()

    average_test_loss = total_test_loss / len(test_dataloader)
    print(f"Average Test Loss: {average_test_loss:.4f}")

# Save the best model
torch.save(model.state_dict(), "m.pt")


# model_data = torch.load("m.pt")
#
# shots_bounds = model_data["shots_bounds"]
# scores_within_shots = model_data["scores_within_shots"]
# num_frames_list = model_data["num_frames_list"]
# positions = model_data["positions"]
# fps = model_data["fps"]
#
# summaries, timestamps = generate_summary(
#     shots_bounds, scores_within_shots, num_frames_list, positions, fps
# )
# print(summaries)
# print("*" * 10)
# print(timestamps)
