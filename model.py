import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(torch.nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(LayerNorm, self).__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.bias = torch.nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = (x.var(dim=-1, keepdim=True) + self.eps).sqrt()
        normalized = (x - mean) / std
        return self.weight * normalized + self.bias


class FrameLevelTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(FrameLevelTransformer, self).__init__()

        self.num_heads = num_heads
        self.attention_dim = hidden_dim // num_heads

        self.query_projection = nn.Linear(input_dim, hidden_dim)
        self.key_projection = nn.Linear(input_dim, hidden_dim)
        self.value_projection = nn.Linear(input_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.W_0 = nn.Linear(hidden_dim * num_heads, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: input frame features (batch_size, num_frames, input_dim)

        batch_size, num_frames, _ = x.size()

        queries = self.query_projection(x)
        keys = self.key_projection(x)
        values = self.value_projection(x)

        queries = queries.view(
            batch_size * num_frames, self.num_heads, self.attention_dim
        )
        keys = keys.view(batch_size * num_frames, self.num_heads, self.attention_dim)
        values = values.view(
            batch_size * num_frames, self.num_heads, self.attention_dim
        )

        # Compute attention scores
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1))
        attention_scores = attention_scores / (self.attention_dim**0.5)
        attention_weights = torch.softmax(attention_scores, dim=-1)

        # Apply attention to values
        context = torch.matmul(attention_weights, values)
        context = context.view(batch_size, num_frames, -1)

        # Apply linear layers with layer normalization
        x = self.layer_norm(context)
        x = self.linear(x)
        x = self.relu(x)

        # Reduce concatenated dim
        x = x.view(batch_size, num_frames, self.num_heads * self.attention_dim)
        x = self.W_0(x)

        return x


class ShotLevelTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads):
        super(ShotLevelTransformer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(input_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, features):
        features = features.transpose(0, 1)

        # Self-attention
        attention_output, _ = self.multihead_attention(features, features, features)

        attention_output = self.layer_norm1(features + attention_output)
        feed_forward_output = self.feed_forward(attention_output)
        shot_representation = self.layer_norm2(attention_output + feed_forward_output)
        return shot_representation


class AudioVisualSelfAttention(nn.Module):
    def __init__(self, input_dim):
        super(AudioVisualSelfAttention, self).__init__()
        self.query_projection = nn.Linear(input_dim, input_dim)
        self.key_projection = nn.Linear(input_dim, input_dim)
        self.value_projection = nn.Linear(input_dim, input_dim)

    def forward(self, visual_features, audio_features):
        # Project the query, keys, and values
        query = self.query_projection(visual_features)
        keys = self.key_projection(audio_features)
        values = self.value_projection(audio_features)

        # Compute attention scores
        scores = torch.matmul(query, keys.transpose(-2, -1))
        attention_weights = F.softmax(scores, dim=-1)

        # Apply attention weights to values
        attended_values = torch.matmul(attention_weights, values)

        return attended_values


class VideoSummarizationModel(nn.Module):
    def __init__(self, visual_input_dim, hidden_dim, num_heads):
        super(VideoSummarizationModel, self).__init__()
        self.frame_level_transformer = FrameLevelTransformer(
            visual_input_dim, hidden_dim, num_heads
        )
        self.audio_guided_self_attention = AudioVisualSelfAttention(visual_input_dim)
        self.shot_level_transformer = ShotLevelTransformer(
            visual_input_dim, hidden_dim, num_heads
        )
        self.probability_layer = nn.Linear(visual_input_dim, 1)

    def forward(self, visual_features, audio_features, shots_boundaries):
        # Frame-level transformer
        frame_representations = self.frame_level_transformer(visual_features)

        # Shot-level transformer
        shot_representations = []
        for shot_boundary in shots_boundaries:
            shot_frames = frame_representations[shot_boundary[0] : shot_boundary[1]]
            shot_representation = self.shot_level_transformer(shot_frames)
            shot_representations.append(shot_representation)

        # Concatenate shot representations
        combined_features = torch.cat(shot_representations, dim=0)

        # Audio-visual self-attention
        attended_audio_features = self.audio_guided_self_attention(
            combined_features, audio_features
        )

        # Concatenate visual and audio features
        combined_features = torch.cat(
            (combined_features, attended_audio_features), dim=-1
        )

        # Calculate probability of including each shot in the summary
        probabilities = F.softmax(self.probability_layer(combined_features), dim=0)

        return probabilities
