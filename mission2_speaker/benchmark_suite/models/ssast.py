import torch
import torch.nn as nn

class PatchEmbedAudio(nn.Module):
    """
    오디오 전용 패치 임베딩.
    speaker_diarization_analysis.md: '주파수 128 bins, 시간축 단위로 비대칭 패치 구성'
    """
    def __init__(self, in_channels=1, embed_dim=192, patch_f=16, patch_t=16):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=(patch_f, patch_t), stride=(patch_f, patch_t))

    def forward(self, x):
        # x: (B, 1, F=128, T)
        x = self.proj(x) # (B, embed_dim, F_p, T_p)
        x = x.flatten(2).transpose(1, 2) # (B, num_patches, embed_dim)
        return x

class SSAST_Tiny(nn.Module):
    """
    [분석 문서 비교 기준선 2] SSAST-Tiny 경량 오디오 트랜스포머 (~6M 파라미터).
    ViT 가설 검증용 대조군.
    """
    def __init__(self, num_classes=2, embed_dim=192, depth=6, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbedAudio(in_channels=1, embed_dim=embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 트랜스포머 인코더 레이어
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=0.1,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(embed_dim, num_classes)
        )

    def forward(self, x):
        # x: (B, 1, F, T)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.shape[0]

        tokens = self.patch_embed(x) # (B, N, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat((cls_tokens, tokens), dim=1)

        # 트랜스포머 통과
        feat = self.transformer(x_tok)
        cls_feat = self.norm(feat[:, 0]) # CLS 토큰 추출

        logits = self.classifier(cls_feat)
        return logits

    def get_embedding(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        B = x.shape[0]
        tokens = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_tok = torch.cat((cls_tokens, tokens), dim=1)
        feat = self.transformer(x_tok)
        return self.norm(feat[:, 0])
