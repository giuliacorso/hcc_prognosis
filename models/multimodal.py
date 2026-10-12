import torch
import torch.nn as nn
import torch.nn.functional as F


class Multimodal(nn.Module):
    """
    WSI + Clinical 
    """
    def __init__(
        self,
        wsi_input_dim=1536,
        wsi_inner_dim=512,
        clinical_dim=20,
        clinical_emb_dim=64,
        num_latent_queries=2,
        output_dim=2,
        dropout=0.5,
        input_modalities = ["WSI"],
        fuse_modality = "concat"
    ):
        super().__init__()

        # -------------------------
        # WSI branch
        # -------------------------
        self.wsi_proj = nn.Linear(wsi_input_dim, wsi_inner_dim)
        self.latent_queries = nn.Parameter(
            torch.randn(num_latent_queries, wsi_inner_dim)
        )

        self.W_k = nn.Linear(wsi_inner_dim, wsi_inner_dim)
        self.gate = nn.Linear(wsi_inner_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.fc_wsi = nn.Linear(num_latent_queries * wsi_inner_dim, wsi_inner_dim)

        # -------------------------
        # Clinical branch
        # -------------------------
        self.clinical_encoder = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(clinical_dim, clinical_emb_dim),
            nn.ReLU(), 
            nn.Linear(clinical_emb_dim, clinical_emb_dim)
        )

        # -------------------------
        # Fusion + Classifier
        # -------------------------
        if fuse_modality not in ("concat", "sum"):
            raise ValueError(f"Unsupported fuse_modality: {fuse_modality}")
        self.fuse_modality = fuse_modality

        if fuse_modality == "concat":
            fusion_dim = 0
            if "WSI" in input_modalities:
                fusion_dim += wsi_inner_dim
            if "Clinical" in input_modalities:
                fusion_dim += clinical_emb_dim
        
        if fuse_modality == "sum":
            fusion_dim = wsi_inner_dim

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(fusion_dim, output_dim)

    def forward(self, patch_features, clinical_features=None):
        """
        patch_features: (n_patches, 1536)
        clinical_features: (n_features,)
        """

        # ---------- WSI ----------
        x = self.wsi_proj(patch_features).unsqueeze(0)  # (1, K, 512)
        gate = self.sigmoid(self.gate(x))

        keys = self.W_k(x)
        scores = torch.matmul(
            self.latent_queries, keys.transpose(1, 2)
        ) / torch.sqrt(torch.tensor(keys.size(-1)).float())

        scores = gate.transpose(-1, -2) * scores
        A = F.softmax(scores, dim=-1)

        latent = torch.matmul(A, x).flatten(start_dim=1)
        wsi_embedding = self.fc_wsi(latent)

        # ---------- Clinical ----------
        if clinical_features is not None:
            clinical_embedding = self.clinical_encoder(clinical_features)
            if self.fuse_modality == "concat":
                fused = torch.cat([wsi_embedding, clinical_embedding], dim=1)
            if self.fuse_modality == "sum":
                fused = wsi_embedding + clinical_embedding
        else:
            fused = wsi_embedding

        fused = self.dropout(fused)
        logits = self.classifier(fused)
        Y_prob = F.softmax(logits, dim=1)
        Y_hat = torch.topk(logits, 1, dim=1)[1]

        return logits, Y_prob, Y_hat
