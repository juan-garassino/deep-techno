"""
Vision encoder for sheet music → MIDI (Phase 2 / AUD-partiture-to-midi).

Encodes sheet music images into encoder hidden states that the Music Transformer
decoder attends to via cross-attention.

Dataset target: PrIMuS (87k PNG + agnostic encoding pairs)
  https://grfia.dlsi.ua.es/primus/

Usage:
    encoder = VisionEncoder(d_model=512)
    image   = torch.randn(1, 3, 224, 224)   # (B, C, H, W)
    memory  = encoder(image)                 # (B, num_patches, d_model)
    # Feed memory as cross-attention context to MusicTransformer decoder.

NOTE: this module is a placeholder scaffold. The full ViT backbone and
cross-attention wiring into the decoder are Phase 2 work.
"""
import torch
import torch.nn as nn


class VisionEncoder(nn.Module):
    """
    Patch-based vision encoder for sheet music images.

    Uses a lightweight ViT-style patch projection. For production, swap
    the patch_proj for a pretrained ViT backbone via timm:
        import timm
        self.backbone = timm.create_model("vit_small_patch16_224", pretrained=True, num_classes=0)
    """

    def __init__(self, d_model: int = 512,
                 img_size = 224,
                 patch_size: int = 16, in_channels: int = 3, dropout: float = 0.1):
        super().__init__()
        self.patch_size = patch_size
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        h_patches = img_size[0] // patch_size
        w_patches = img_size[1] // patch_size
        num_patches = h_patches * w_patches
        patch_dim = in_channels * patch_size * patch_size

        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, d_model))
        self.dropout = nn.Dropout(dropout)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) float tensor, values in [0, 1]
        Returns:
            (B, num_patches, d_model) encoder memory
        """
        B, C, H, W = images.shape
        p = self.patch_size
        patches = images.unfold(2, p, p).unfold(3, p, p)           # (B, C, nH, nW, p, p)
        patches = patches.contiguous().view(B, C, -1, p * p)       # (B, C, N, p²)
        patches = patches.permute(0, 2, 1, 3).flatten(2)           # (B, N, C*p²)
        x = self.patch_proj(patches) + self.pos_embedding
        return self.dropout(x)
