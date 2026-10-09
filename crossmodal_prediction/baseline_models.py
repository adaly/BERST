import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GATv2Conv


#########################################################################################
##                                   MLP MODEL BLOCKS                                  ##
#########################################################################################

class ImageMLPEncoder(nn.Module):
    """
    Image MLP with optional metadata integration and optional
    gene-expression prediction head.

    metadata_mode:
        "none"   : image features only
        "tissue" : image features + tissue embedding
        "all"    : image features + all metadata

    If n_genes is None:
        returns [B, latent_dim] latent representation.

    If n_genes is supplied:
        returns [B, n_genes] expression predictions.
    """

    def __init__(
        self,
        input_dim,
        n_genes=None,
        hidden_dim=1024,
        latent_dim=256,
        n_blocks=2,
        dropout=0.2,
        metadata_mode="none",

        # Metadata configuration
        categorical_cardinalities=None,
        tissue_index=1,
        categorical_embedding_dim=8,
        tissue_embedding_dim=16,
        n_continuous=3,
        metadata_hidden_dim=64,
    ):
        super().__init__()

        if n_blocks < 0:
            raise ValueError("n_blocks must be >= 0")

        if metadata_mode not in {"none", "tissue", "all"}:
            raise ValueError(
                "metadata_mode must be one of "
                "{'none', 'tissue', 'all'}"
            )

        self.metadata_mode = metadata_mode
        self.tissue_index = tissue_index
        self.latent_dim = latent_dim

        # ----------------------------------------------------------
        # Image encoder
        # ----------------------------------------------------------

        layers = [nn.LayerNorm(input_dim)]
        current_dim = input_dim

        for _ in range(n_blocks):
            layers.extend([
                nn.Linear(current_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            current_dim = hidden_dim

        layers.extend([
            nn.Linear(current_dim, latent_dim),
            nn.GELU(),
        ])

        self.image_encoder = nn.Sequential(*layers)

        # ----------------------------------------------------------
        # Metadata components
        # ----------------------------------------------------------

        if metadata_mode != "none":

            if categorical_cardinalities is None:
                raise ValueError(
                    "categorical_cardinalities is required "
                    "when using metadata."
                )

            if metadata_mode == "tissue":

                n_tissues = categorical_cardinalities[
                    tissue_index
                ]

                self.tissue_embedding = nn.Embedding(
                    n_tissues,
                    tissue_embedding_dim,
                )

                metadata_input_dim = tissue_embedding_dim

            else:  # metadata_mode == "all"

                self.categorical_embeddings = nn.ModuleList([
                    nn.Embedding(
                        n_categories,
                        categorical_embedding_dim,
                    )
                    for n_categories
                    in categorical_cardinalities
                ])

                metadata_input_dim = (
                    len(categorical_cardinalities)
                    * categorical_embedding_dim
                    + 2 * n_continuous
                )

            self.metadata_encoder = nn.Sequential(
                nn.Linear(
                    metadata_input_dim,
                    metadata_hidden_dim,
                ),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(
                    metadata_hidden_dim,
                    metadata_hidden_dim,
                ),
                nn.GELU(),
            )

            # Fuse image + metadata and return to latent_dim.
            self.fusion = nn.Sequential(
                nn.Linear(
                    latent_dim + metadata_hidden_dim,
                    latent_dim,
                ),
                nn.GELU(),
            )

        # ----------------------------------------------------------
        # Optional expression prediction head
        # ----------------------------------------------------------

        if n_genes is not None:
            self.expression_head = nn.Linear(
                latent_dim,
                n_genes,
            )
        else:
            self.expression_head = None

    def forward(
        self,
        x,
        categorical_metadata=None,
        continuous_metadata=None,
        continuous_metadata_mask=None,
        tissue_id=None,
    ):

        # ----------------------------------------------------------
        # Image representation
        # ----------------------------------------------------------

        z = self.image_encoder(x)

        # ----------------------------------------------------------
        # Metadata representation
        # ----------------------------------------------------------

        if self.metadata_mode != "none":

            if self.metadata_mode == "tissue":

                if tissue_id is None:
                    if categorical_metadata is None:
                        raise ValueError(
                            "tissue_id or categorical_metadata "
                            "must be supplied."
                        )

                    tissue_id = categorical_metadata[
                        ..., self.tissue_index
                    ]

                z_metadata = self.tissue_embedding(
                    tissue_id.long()
                )

            else:  # metadata_mode == "all"

                if categorical_metadata is None:
                    raise ValueError(
                        "categorical_metadata is required."
                    )

                if continuous_metadata is None:
                    raise ValueError(
                        "continuous_metadata is required."
                    )

                if continuous_metadata_mask is None:
                    raise ValueError(
                        "continuous_metadata_mask is required."
                    )

                categorical_embeddings = [
                    embedding(
                        categorical_metadata[..., i].long()
                    )
                    for i, embedding
                    in enumerate(
                        self.categorical_embeddings
                    )
                ]

                z_metadata = torch.cat(
                    categorical_embeddings
                    + [
                        continuous_metadata.float(),
                        continuous_metadata_mask.float(),
                    ],
                    dim=-1,
                )

            z_metadata = self.metadata_encoder(
                z_metadata
            )

            z = torch.cat(
                [z, z_metadata],
                dim=-1,
            )

            z = self.fusion(z)

        # ----------------------------------------------------------
        # Optional prediction
        # ----------------------------------------------------------

        if self.expression_head is not None:
            z = self.expression_head(z)

        return z


#########################################################################################
##                                 SPATIAL MODEL BLOCKS                                ##
#########################################################################################

class TissueFiLM(nn.Module):
    """
    Generate feature-wise affine modulation parameters from tissue identity.

    tissue_id should contain one tissue index per node after any graph-level
    tissue IDs have been expanded/mapped to nodes.
    """

    def __init__(
        self,
        n_tissues,
        latent_dim,
        tissue_embedding_dim=16,
    ):
        super().__init__()

        self.tissue_embedding = nn.Embedding(
            n_tissues,
            tissue_embedding_dim,
        )

        self.film = nn.Linear(
            tissue_embedding_dim,
            2 * latent_dim,
        )

        # Start close to identity modulation:
        # gamma = beta = 0 initially.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x, tissue_id):
        tissue_z = self.tissue_embedding(
            tissue_id.long()
        )

        gamma, beta = self.film(
            tissue_z
        ).chunk(2, dim=-1)

        return (1.0 + gamma) * x + beta

        
class ResidualGCNBlock(nn.Module):

    def __init__(
        self,
        latent_dim,
        dropout=0.2,
        tissue_conditioning=False,
        n_tissues=None,
        tissue_embedding_dim=16,
    ):
        super().__init__()

        self.tissue_conditioning = tissue_conditioning
        
        self.norm = nn.LayerNorm(
            latent_dim
        )

        self.conv = GCNConv(
            latent_dim,
            latent_dim,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        if tissue_conditioning:

            if n_tissues is None:
                raise ValueError(
                    "n_tissues is required when "
                    "tissue_conditioning=True."
                )

            self.tissue_film = TissueFiLM(
                n_tissues=n_tissues,
                latent_dim=latent_dim,
                tissue_embedding_dim=(
                    tissue_embedding_dim
                ),
            )

    def forward(
        self,
        x,
        edge_index,
        tissue_id=None,
    ):

        residual = x

        x = self.norm(x)

        # Optionally, apply FiLM conditioning (based on tissue type label)
        if self.tissue_conditioning:

            if tissue_id is None:
                raise ValueError(
                    "tissue_id must be supplied when "
                    "tissue conditioning is enabled."
                )

            x = self.tissue_film(
                x,
                tissue_id,
            )

        x = self.conv(
            x,
            edge_index,
        )

        x = self.activation(x)

        x = self.dropout(x)

        return residual + x

class ResidualGATBlock(nn.Module):

    def __init__(
        self,
        latent_dim,
        heads=4,
        dropout=0.2,
        tissue_conditioning=False,
        n_tissues=None,
        tissue_embedding_dim=16,
    ):
        super().__init__()

        if latent_dim % heads != 0:
            raise ValueError(
                "latent_dim must be divisible "
                "by heads."
            )

        self.tissue_conditioning = tissue_conditioning

        self.norm = nn.LayerNorm(
            latent_dim
        )

        self.attention = GATv2Conv(
            in_channels=latent_dim,
            out_channels=latent_dim // heads,
            heads=heads,
            concat=True,
            dropout=dropout,
        )

        self.activation = nn.GELU()

        self.dropout = nn.Dropout(
            dropout
        )

        if tissue_conditioning:

            if n_tissues is None:
                raise ValueError(
                    "n_tissues is required when "
                    "tissue_conditioning=True."
                )

            self.tissue_film = TissueFiLM(
                n_tissues=n_tissues,
                latent_dim=latent_dim,
                tissue_embedding_dim=(
                    tissue_embedding_dim
                ),
            )

    def forward(
        self,
        x,
        edge_index,
        tissue_id=None,
    ):

        residual = x

        x = self.norm(x)

        if self.tissue_conditioning:

            if tissue_id is None:
                raise ValueError(
                    "tissue_id must be supplied when "
                    "tissue conditioning is enabled."
                )

            x = self.tissue_film(
                x,
                tissue_id,
            )

        x = self.attention(
            x,
            edge_index,
        )

        x = self.activation(x)

        x = self.dropout(x)

        return residual + x


class SpatialMetadataMixin:

    @staticmethod
    def _to_nodes(value, x, batch=None):
        """
        Convert graph-level metadata [B, ...] to node-level
        metadata [N, ...].

        For batch_size=1:
            [1, M] -> [N, M]

        For batched PyG graphs:
            [B, M] -> [N, M] using batch[node] -> graph.
        """

        if value is None:
            return None

        # Scalar -> [1]
        if value.ndim == 0:
            value = value.unsqueeze(0)

        if batch is not None:
            # Already node-level.
            if value.shape[0] == x.shape[0]:
                return value

            # Graph-level -> node-level.
            return value[batch]

        # No PyG batch vector: assume one graph.
        if value.shape[0] == x.shape[0]:
            return value

        if value.shape[0] == 1:
            return value.expand(
                x.shape[0],
                *value.shape[1:],
            )

        raise ValueError(
            "Could not map metadata to nodes: "
            f"x has {x.shape[0]} nodes but metadata "
            f"has leading dimension {value.shape[0]}."
        )

    def _expand_metadata(
        self,
        x,
        categorical_metadata=None,
        continuous_metadata=None,
        continuous_metadata_mask=None,
        tissue_id=None,
        batch=None,
    ):

        return {
            "categorical_metadata": self._to_nodes(
                categorical_metadata,
                x,
                batch,
            ),
            "continuous_metadata": self._to_nodes(
                continuous_metadata,
                x,
                batch,
            ),
            "continuous_metadata_mask": self._to_nodes(
                continuous_metadata_mask,
                x,
                batch,
            ),
            "tissue_id": self._to_nodes(
                tissue_id,
                x,
                batch,
            ),
        }

#########################################################################################
##                                    RESIDUAL GCN                                     ##
#########################################################################################

class SpatialGCNExpressionModel(SpatialMetadataMixin, nn.Module):
    """
    Spatial1 benchmark.

    Virchow2
        -> MLP encoder
        -> N residual GCN blocks
        -> expression head

    Parameters
    ----------
    input_dim : int
        Virchow2 feature dimension.

    n_genes : int
        Number of expression targets.

    n_spatial_blocks : int
        Number of GCN blocks. This is the primary
        spatial-complexity parameter.

    n_mlp_blocks : int
        Number of blocks in the image encoder.

    hidden_dim : int
        MLP hidden width.

    latent_dim : int
        Dimension used by the spatial model.

    dropout : float
        Dropout probability.
    """

    def __init__(
        self,
        input_dim,
        n_genes,
        n_spatial_blocks=1,
        n_mlp_blocks=2,
        hidden_dim=1024,
        latent_dim=256,
        dropout=0.2,

        # Metadata used by ImageMLPEncoder
        metadata_mode="none",
        categorical_cardinalities=None,
        tissue_index=1,
        categorical_embedding_dim=8,
        tissue_embedding_dim=16,
        n_continuous=3,
        metadata_hidden_dim=64,

        # Separate switch controlling whether tissue
        # conditions the spatial layers themselves
        tissue_conditioning=False,
    ):
        super().__init__()

        if n_spatial_blocks < 0:
            raise ValueError(
                "n_spatial_blocks must be >= 0"
            )

        self.tissue_conditioning = tissue_conditioning

        # Number of tissue categories, including "unknown"
        n_tissues = None

        if tissue_conditioning:

            if categorical_cardinalities is None:
                raise ValueError(
                    "categorical_cardinalities is required "
                    "for tissue conditioning."
                )

            n_tissues = (
                categorical_cardinalities[
                    tissue_index
                ]
            )

        # --------------------------------------------------
        # Image + optional generic metadata encoder
        # --------------------------------------------------

        self.image_encoder = ImageMLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_blocks=n_mlp_blocks,
            dropout=dropout,

            metadata_mode=metadata_mode,
            categorical_cardinalities=(
                categorical_cardinalities
            ),
            tissue_index=tissue_index,
            categorical_embedding_dim=(
                categorical_embedding_dim
            ),
            tissue_embedding_dim=(
                tissue_embedding_dim
            ),
            n_continuous=n_continuous,
            metadata_hidden_dim=(
                metadata_hidden_dim
            ),
        )

        # --------------------------------------------------
        # Spatial layers
        # --------------------------------------------------
        
        self.spatial_blocks = nn.ModuleList([
            ResidualGCNBlock(
                latent_dim=latent_dim,
                dropout=dropout,
                tissue_conditioning=(
                    tissue_conditioning
                ),
                n_tissues=n_tissues,
                tissue_embedding_dim=(
                    tissue_embedding_dim
                ),
            )
            for _ in range(n_spatial_blocks)
        ])

        self.expression_head = nn.Linear(
            latent_dim,
            n_genes,
        )

    def forward(
        self,
        x,
        edge_index,
        categorical_metadata=None,
        continuous_metadata=None,
        continuous_metadata_mask=None,
        tissue_id=None,
        batch=None,
    ):

        # Convert graph-level metadata to node-level metadata.
        metadata = self._expand_metadata(
            x=x,
            categorical_metadata=categorical_metadata,
            continuous_metadata=continuous_metadata,
            continuous_metadata_mask=(
                continuous_metadata_mask
            ),
            tissue_id=tissue_id,
            batch=batch,
        )
        
        z = self.image_encoder(
            x,
            categorical_metadata=metadata[
                "categorical_metadata"
            ],
            continuous_metadata=metadata[
                "continuous_metadata"
            ],
            continuous_metadata_mask=metadata[
                "continuous_metadata_mask"
            ],
            tissue_id=metadata["tissue_id"],
        )

        for block in self.spatial_blocks:

            z = block(
                z,
                edge_index,
                tissue_id=metadata["tissue_id"],
            )

        return self.expression_head(z)


#########################################################################################
##                                    RESIDUAL GAT                                     ##
#########################################################################################

class SpatialGATExpressionModel(SpatialMetadataMixin, nn.Module):
    """
    Spatial2 benchmark.

    Virchow2
        -> MLP encoder
        -> N residual GATv2 blocks
        -> expression head
    """

    def __init__(
        self,
        input_dim,
        n_genes,
        n_spatial_blocks=1,
        n_mlp_blocks=2,
        hidden_dim=1024,
        latent_dim=256,
        heads=4,
        dropout=0.2,

        # Generic MLP metadata
        metadata_mode="none",
        categorical_cardinalities=None,
        tissue_index=1,
        categorical_embedding_dim=8,
        tissue_embedding_dim=16,
        n_continuous=3,
        metadata_hidden_dim=64,

        # Spatial conditioning
        tissue_conditioning=False,
    ):
        super().__init__()

        if n_spatial_blocks < 0:
            raise ValueError(
                "n_spatial_blocks must be >= 0"
            )

        self.tissue_conditioning = tissue_conditioning

        n_tissues = None

        if tissue_conditioning:

            if categorical_cardinalities is None:
                raise ValueError(
                    "categorical_cardinalities is required "
                    "for tissue conditioning."
                )

            n_tissues = (
                categorical_cardinalities[
                    tissue_index
                ]
            )

        # --------------------------------------------------
        # Image + optional generic metadata encoder
        # --------------------------------------------------

        self.image_encoder = ImageMLPEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            n_blocks=n_mlp_blocks,
            dropout=dropout,

            metadata_mode=metadata_mode,
            categorical_cardinalities=(
                categorical_cardinalities
            ),
            tissue_index=tissue_index,
            categorical_embedding_dim=(
                categorical_embedding_dim
            ),
            tissue_embedding_dim=(
                tissue_embedding_dim
            ),
            n_continuous=n_continuous,
            metadata_hidden_dim=(
                metadata_hidden_dim
            ),
        )

        # --------------------------------------------------
        # Spatial layers
        # --------------------------------------------------
        
        self.spatial_blocks = nn.ModuleList([
            ResidualGATBlock(
                latent_dim=latent_dim,
                heads=heads,
                dropout=dropout,
                tissue_conditioning=(
                    tissue_conditioning
                ),
                n_tissues=n_tissues,
                tissue_embedding_dim=(
                    tissue_embedding_dim
                ),
            )
            for _ in range(n_spatial_blocks)
        ])

        self.expression_head = nn.Linear(
            latent_dim,
            n_genes,
        )

    def forward(
        self,
        x,
        edge_index,
        categorical_metadata=None,
        continuous_metadata=None,
        continuous_metadata_mask=None,
        tissue_id=None,
        batch=None,
    ):

        metadata = self._expand_metadata(
            x=x,
            categorical_metadata=categorical_metadata,
            continuous_metadata=continuous_metadata,
            continuous_metadata_mask=(
                continuous_metadata_mask
            ),
            tissue_id=tissue_id,
            batch=batch,
        )
        
        z = self.image_encoder(
            x,
            categorical_metadata=metadata[
                "categorical_metadata"
            ],
            continuous_metadata=metadata[
                "continuous_metadata"
            ],
            continuous_metadata_mask=metadata[
                "continuous_metadata_mask"
            ],
            tissue_id=metadata["tissue_id"],
        )

        for block in self.spatial_blocks:

            z = block(
                z,
                edge_index,
                tissue_id=metadata["tissue_id"],
            )

        return self.expression_head(z)