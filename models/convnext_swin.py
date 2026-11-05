# Autora: Giulia Katherine Romas Tomazeli

import torch
import torch.nn as nn

from torchvision.models import (convnext_tiny, ConvNeXt_Tiny_Weights, swin_t, Swin_T_Weights)

class AttentionPooling(nn.Module):
    def __init__(self, feature_dim, attention_dim=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, x):
        """
        x:[B, N, C]
        """
        # [B, N, 1]
        attention_scores = self.attention(x)

        # normaliza os pesos espacialmente
        attention_weights = torch.softmax(attention_scores, dim=1)

        # soma ponderada
        # [B, N, C] * [B, N, 1]
        features = torch.sum(attention_weights * x, dim=1)

        # [B, C]
        return features
    
class ConvNeXtSwin(nn.Module):
    def __init__(self, num_classes, pretrained=True, freeze_backbone=True, hidden_dim=512, dropout=0.4):
        super().__init__()

        # -=-=-=-=- convnext -=-=-=-=- #
        convnext_weights = (ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        self.convnext = convnext_tiny(weights=convnext_weights)
        convnext_features = (self.convnext.classifier[-1].in_features)

        # remove o classificador
        self.convnext.classifier[-1] = nn.Identity()

        # -=-=-=-=- swin-tiny -=-=-=-=- #
        swin_weights = (Swin_T_Weights.DEFAULT if pretrained else None)
        self.swin = swin_t(weights=swin_weights)
        swin_features = self.swin.head.in_features

        # remove o classificador
        self.swin.head = nn.Identity()

        # -=-=-=-=- congela backbone -=-=-=-=- #
        if freeze_backbone:
            for param in self.convnext.parameters():
                param.requires_grad = False

            for param in self.swin.parameters():
                param.requires_grad = False

        # -=-=-=-=- attention pooling -=-=-=-=- #
        self.convnext_attention = AttentionPooling(feature_dim=convnext_features)
        self.swin_attention = AttentionPooling(feature_dim=swin_features)

        # -=-=-=-=- normalizacao -=-=-=-=- #
        self.convnext_norm = nn.LayerNorm(convnext_features)
        self.swin_norm = nn.LayerNorm(swin_features)

        # -=-=-=-=- classificador -=-=-=-=- #
        self.classifier = nn.Sequential(
            nn.Linear(convnext_features + swin_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # -=-=-=-=- convnext -=-=-=-=- #
        convnext_features = self.convnext.features(x)
        convnext_features = convnext_features.flatten(2).transpose(1, 2)
        convnext_features = self.convnext_attention(convnext_features)

        # -=-=-=-=- swin -=-=-=-=- #
        swin_features = self.swin.features(x)
        swin_features = swin_features.reshape(swin_features.size(0), -1, swin_features.size(-1))
        swin_features = self.swin_attention(swin_features)

        # -=-=-=-=- fusao -=-=-=-=- #
        convnext_features = self.convnext_norm(convnext_features)
        swin_features = self.swin_norm(swin_features)

        features = torch.cat([convnext_features, swin_features], dim=1)

        # -=-=-=-=- classificador -=-=-=-=- #
        output = self.classifier(features)

        return output

def create_model(num_classes, pretrained=True, freeze_backbone=True, hidden_dim=512, dropout=0.4):
    """
    Cria o modelo híbrido ConvNeXt-Tiny + Swin-Tiny.
    """
    model = ConvNeXtSwin(num_classes=num_classes, pretrained=pretrained, freeze_backbone=freeze_backbone, hidden_dim=hidden_dim, dropout=dropout)
    return model