import numpy as np

import pkgutil
from io import BytesIO

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import is_undirected

from einops import rearrange
from functools import partial
from reversible import GraphSequence
from performer_pytorch import default, cast_tuple, find_modules, get_module_device, exists
from performer_pytorch import FastAttention, SelfAttention, Gene2VecPositionalEmbedding, NaiveAttention, FlashAttention
from performer_pytorch import PreLayerNorm, PreScaleNorm, ReZero, Chunk, Always, FeedForward

###################################################################################################
# Graph-based cross-attention:                                                                    #
# - Operates on nodes with 2D features (tokens x token_dim)                                       #
# - For a given node, computes average K, V matrices across neighbors                             #
###################################################################################################

class CrossAttention(MessagePassing):
    def __init__(
        self,
        dim,
        causal = False,
        heads = 8,
        dim_head = 64,
        nb_features = None,
        feature_redraw_interval = 1000,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        dropout = 0.,
        no_projection = False,
        qkv_bias = False,
        fast_attention = True,
        attn_chunksize = None
    ):
        super().__init__(aggr="mean", flow="source_to_target")
        
        assert dim % heads == 0, 'dimension must be divisible by number of heads'
        dim_head = default(dim_head, dim // heads)
        self.inner_dim = dim_head * heads  # multi-head attention achieved by concatenating multiple weight matrices along feature dimension
        
        self.heads = heads
        if fast_attention:
            #self.fast_attention = FastAttention(dim_head, nb_features, causal = causal, generalized_attention = generalized_attention, kernel_fn = kernel_fn, no_projection = no_projection)
            self.fast_attention = FlashAttention(causal = causal)
        else:
            self.fast_attention = NaiveAttention()   # for debugging/testing
        if attn_chunksize is not None:
            assert isinstance(attn_chunksize, int), 'attn_chunksize must be integer valued if provided'
        self.attn_chunksize = attn_chunksize

        self.to_q = nn.Linear(dim, self.inner_dim, bias = qkv_bias)
        self.to_k = nn.Linear(dim, self.inner_dim, bias = qkv_bias)
        self.to_v = nn.Linear(dim, self.inner_dim, bias = qkv_bias)
        self.to_out = nn.Linear(self.inner_dim, dim)
        self.dropout = nn.Dropout(dropout)

    # Calculates FastAttention separately for each node in order to save GPU memory
    def fast_attention_chunked(self, q, k, v, output_attentions=False):
        if self.attn_chunksize is None:
            return self.fast_attention(q, k, v, output_attentions)
            
        n_nodes = q.shape[0]
        if n_nodes == 1:
            return self.fast_attention(q, k, v, output_attentions=output_attentions)

        n_chunks = int(np.ceil(float(q.shape[0]) / self.attn_chunksize))   # number of chunks of size attn_chunksize
        q_chunks = q.chunk(n_chunks, dim = 0)
        k_chunks = k.chunk(n_chunks, dim = 0)
        v_chunks = v.chunk(n_chunks, dim = 0)
        attn_chunks = [self.fast_attention(qc, kc, vc, output_attentions=output_attentions) for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks)]
        
        if output_attentions:
            # Separately combine chunked outputs and attention weights
            out = torch.cat([c[0] for c in attn_chunks], dim=0)
            attn_weights = torch.cat([c[1] for c in attn_chunks], dim=0)
            return out, attn_weights
        else:
            return torch.cat(attn_chunks, dim = 0)

    # Defines what is sent from source (x_j) to target (x_i) nodes
    # TODO: add optional argument for edge features (e.g., distance-based weights)
    def message(self, x_i, x_j, n_tokens, token_dim, output_attentions=False):
        n_edges = x_i.shape[0]

        # Reshape x_i and x_j to separate tokens & embedding (token_dim):
        x_tgt = x_i.view(n_edges, n_tokens, token_dim)
        x_src = x_j.view(n_edges, n_tokens, token_dim)

        #print('x_tgt:', x_tgt.shape)   # (n_edges, n_tokens, token_dim)
        #print('x_src:', x_src.shape)   # (n_edges, n_tokens, token_dim)

        # Calculate K, V from SOURCE node
        k, v = self.to_k(x_src), self.to_v(x_src)

        # Calculate Q from TARGET node
        q = self.to_q(x_tgt)

        # Separate heads into a new dimension
        b, n = n_edges, n_tokens
        h = self.heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), (q, k, v))

        # Calculate attention
        if output_attentions:
            out, attn_weights = self.fast_attention_chunked(q, k, v, output_attentions)
        else:
            out = self.fast_attention_chunked(q, k, v)

        # out.shape: (n_edges, heads, n_tokens, dim_head)

        # Subsume head dimension (as expected by to_out Linear layer)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Flatten for message passing
        out_flat = out.reshape(n_edges, -1)

        # Tack on (flattened) attention weights, if requested
        if output_attentions:
            attn_weights_flat = attn_weights.view(n_edges, -1)
            out_flat = torch.cat((out_flat, attn_weights_flat), -1)

        return out_flat

    def forward(self, x, edge_index, output_attentions = False, target_nodes = None,  **kwargs):
        assert x.ndim == 3  # [n_nodes, n_tokens, token_dim]
        n_nodes, n_tokens, token_dim, h = *x.shape, self.heads

        # Flatten tokens & token dim together for message passing (PyG expects 1d node features)
        x_flat = x.view(n_nodes, n_tokens * token_dim)

        # Receive aggregated update (flattened)
        out_flat = self.propagate(edge_index, x=x_flat, n_tokens=n_tokens, token_dim=token_dim, 
            output_attentions=output_attentions)

        # Separate & reshape attention weights, if provided!
        if output_attentions:
            # First (n_tokens * n_heads * dim_head) variables are node features
            # ...following (n_tokens * n_tokens) variables are attention weights
            out_flat, attn_weights_flat = torch.split(out_flat, [n_tokens*self.inner_dim, n_tokens**2], dim=1)
            attn_weights = attn_weights_flat.view(n_nodes, n_tokens, n_tokens)

        # Reshape to (n_nodes, n_tokens, n_heads * dim_head)
        out = out_flat.view(n_nodes, n_tokens, -1)
        # ...then pass through final linear layer to project back to token_dim!
        out = self.to_out(out.to(torch.float32))

        if target_nodes is not None:
            out = out[target_nodes]
            if output_attentions:
                attn_weights = attn_weights[target_nodes]

        if output_attentions:
            return self.dropout(out), attn_weights
        else:
            return self.dropout(out)

###################################################################################################
# WRAPPER CLASSES:                                                                                #
# - BERST has two types of attention: self (within node) and cross (between nodes)                #
# - Cross-attention requires an additional argument to forward() -- edge_index                    #
# - Rather than making an explicit argument, wrap both forward() methods to accept *args          #
# - Makes things modular; same forward() call be be applied in a Sequential network               #
###################################################################################################

class GraphSelfAttention(SelfAttention):
    def forward(self, x, *args, **kwargs):
        return super(GraphSelfAttention, self).forward(x, **kwargs)

class GraphCrossAttention(CrossAttention):
    def forward(self, x, *args, **kwargs):
        if len(args) == 0:
            raise ValueError("GraphCrossAttention expects 'edge_index' as a second argument")
        edge_index = args[0]
        return super(GraphCrossAttention, self).forward(x, edge_index, **kwargs)
        

###################################################################################################
# scBERT-based implementation of language models                                                  #
# - Alternately apply SelfAttention and CrossAttention up to desired depth                        #
# - ...with token-level FeedForward layers to provide post-processing between Attention layers    #
###################################################################################################

class GraphPerformer(nn.Module):
    def __init__(
        self,
        dim,                                # dimension
        depth,                              # layers
        heads,                              # heads
        dim_head,                           # dim of head
        causal = False,                     # autoregressive or not
        ff_mult = 4,                        # dim of intermediate features after attention / dim of input features
        nb_features = None,                 # number of random features, if not set, will default to (d * log(d)), where d is the dimension of each head ?? what is random feature ??
        feature_redraw_interval = 1000,     # how frequently to redraw the projection matrix, the more frequent, the slower the training
        reversible = False,                 # reversible layers, from Reformer (save memory)
        ff_chunks = 1,                      # chunk feedforward layer, from Reformer
        generalized_attention = False,      # defaults to softmax approximation, but can be set to True for generalized attention ?? what is generalized attention ??
        kernel_fn = nn.ReLU(),              # the kernel function to be used, if generalized attention is turned on, defaults to Relu
        use_scalenorm = False,              # use scale norm, from 'Transformers without Tears' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        use_rezero = False,                 # use Rezero or not, from 'Rezero is all you need' paper, a substitute for LayerNorm, priority: scalenorm.rezero.layernorm
        ff_glu = False,                     # use GLU (Gated Linear Units) variant for feedforward
        ff_dropout = 0.,                    # feedforward dropout
        attn_dropout = 0.,                  # post-attention dropout
        no_projection = False,              # ??
        auto_check_redraw = True,           # ??
        qkv_bias = True,                    # ??
        attn_chunksize = 1                  # number of nodes over which to calculate (cross) attention at a time -- larger = faster but more GPU intensive
    ):
        super().__init__()
        layers = nn.ModuleList([])

        if use_scalenorm:
            wrapper_fn = partial(PreScaleNorm, dim)
        elif use_rezero:
            wrapper_fn = ReZero
        else:
            wrapper_fn = partial(PreLayerNorm, dim)

        for _ in range(depth):
            # Self-attention (within nodes), followed by token-level MLP (FeedForward)
            layers.append(nn.ModuleList([
                wrapper_fn(GraphSelfAttention(dim, causal = causal, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias)),
                wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

            # Cross-attention (between neighbors), followed by token-level MLP (FeedForward)
            layers.append(nn.ModuleList([
            	wrapper_fn(GraphCrossAttention(dim, causal = causal, heads = heads, dim_head = dim_head, nb_features = nb_features, generalized_attention = generalized_attention, kernel_fn = kernel_fn, dropout = attn_dropout, no_projection = no_projection, qkv_bias = qkv_bias, attn_chunksize = attn_chunksize)),
            	wrapper_fn(Chunk(ff_chunks, FeedForward(dim, mult = ff_mult, dropout = ff_dropout, glu = ff_glu), along_dim = 1))
            ]))

        # Sequence classes handle routing of arguments & attention weights through layers 
        # (and apply a residual connection after each layer -- e.g., x = x + f(x))
        if reversible:
            raise NotImplementedError("Have not implemented Reversible layers for graph attention")
        else:
            execute_type = GraphSequence

        route_attn = ((True, False),) * depth * 2                        # True for Attention-based layers, False for FeedForward
        attn_route_map = {'mask': route_attn, 'pos_emb': route_attn}     # kwargs to route to all Attention-based layers
        graph_layers = [False, True] * depth                             # True for layers involving (graph) cross-attention
        self.net = execute_type(layers, graph_layers, args_route = {**attn_route_map})

        # keeping track of when to redraw projections for all attention layers
        self.auto_check_redraw = auto_check_redraw
        self.feature_redraw_interval = feature_redraw_interval
        self.register_buffer('calls_since_last_redraw', torch.tensor(0))

    def fix_projection_matrices_(self):
        self.feature_redraw_interval = None

    def check_redraw_projections(self):
        if not self.training:
            return

        if exists(self.feature_redraw_interval) and self.calls_since_last_redraw >= self.feature_redraw_interval:
            device = get_module_device(self)

            fast_attentions = find_modules(self, FastAttention)
            for fast_attention in fast_attentions:
                fast_attention.redraw_projection_matrix(device)

            self.calls_since_last_redraw.zero_()
            return

        self.calls_since_last_redraw += 1

    def forward(self, x, *args, output_attentions = False, **kwargs):
        if self.auto_check_redraw:
            self.check_redraw_projections()
        return self.net(x, *args, output_attentions = output_attentions, **kwargs)

class GraphPerformerLM(nn.Module):
    def __init__(
        self,
        *,
        num_tokens,                         # num of tokens
        max_seq_len,                        # max length of sequence
        dim,                                # dim of tokens
        depth,                              # layers
        heads,                              # num of heads
        dim_head = 64,                      # dim of heads
        local_attn_heads = 0,
        local_window_size = 256,
        causal = False,
        ff_mult = 4,
        nb_features = None,
        feature_redraw_interval = 1000,
        reversible = False,
        ff_chunks = 1,
        ff_glu = False,
        emb_dropout = 0.,
        ff_dropout = 0.,
        attn_dropout = 0.,
        generalized_attention = False,
        kernel_fn = nn.ReLU(),
        use_scalenorm = False,
        use_rezero = False,
        cross_attend = False,
        no_projection = False,
        tie_embed = False,                  # False: output is num of tokens, True: output is dim of tokens  //multiply final embeddings with token weights for logits, like gpt decoder//
        g2v_position_emb = True,            # priority: gene2vec, no embedding
        auto_check_redraw = True,
        qkv_bias = False,
        attn_chunksize = 1
    ):
        super().__init__()
        local_attn_heads = cast_tuple(local_attn_heads)

        self.max_seq_len = max_seq_len
        self.token_emb = nn.Embedding(num_tokens, dim)

        if g2v_position_emb:
            self.pos_emb = Gene2VecPositionalEmbedding(dim, max_seq_len)
            self.layer_pos_emb = Always(None)
        else:
            self.pos_emb = torch.zeros_like
            self.layer_pos_emb = Always(None)

        self.dropout = nn.Dropout(emb_dropout)

        self.performer = GraphPerformer(dim, depth, heads, dim_head, causal, ff_mult, nb_features, feature_redraw_interval, reversible, ff_chunks, generalized_attention, kernel_fn, use_scalenorm, use_rezero, ff_glu, ff_dropout, attn_dropout, no_projection, auto_check_redraw, qkv_bias, attn_chunksize)
        self.norm = nn.LayerNorm(dim)
        self.to_out = nn.Linear(dim, num_tokens) if not tie_embed else None

    def check_redraw_projections(self):
        self.performer.check_redraw_projections()

    def fix_projection_matrices_(self):
        self.performer.fix_projection_matrices_()

    def forward(self, x, *args, return_encodings = False, output_attentions = False, **kwargs):
        b, n, device = *x.shape, x.device
        assert n <= self.max_seq_len, f'sequence length {n} must be less than the max sequence length {self.max_seq_len}'

        # token and positional embedding
        x = self.token_emb(x)
        if output_attentions:
            x.requires_grad_()    # used for attn_map output
        x += self.pos_emb(x)
        x = self.dropout(x)

        # performer layers
        layer_pos_emb = self.layer_pos_emb(x)

        if output_attentions:
            x, attn_weights = self.performer(x, *args, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)
            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x, attn_weights

            if exists(self.to_out):
                return self.to_out(x), attn_weights

            return (x @ self.token_emb.weight.t()), attn_weights
        else:
            x = self.performer(x, *args, pos_emb = layer_pos_emb, output_attentions = output_attentions, **kwargs)

            # norm and to logits
            x = self.norm(x)
            if return_encodings:
                return x

            if exists(self.to_out):
                x = self.to_out(x)
                return x

            return x @ self.token_emb.weight.t()