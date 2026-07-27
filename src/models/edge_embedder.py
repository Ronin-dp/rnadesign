import torch
import torch.nn.functional as F
from torch import nn
from src.models import utils


class EdgeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(EdgeEmbedder, self).__init__()
        self._cfg = module_cfg

        self.c_s = self._cfg.c_s
        self.c_p = self._cfg.c_p
        self.feat_dim = self._cfg.feat_dim
        self.max_relpos = getattr(self._cfg, "max_relpos", 32)

        self.linear_s_p = nn.Linear(self.c_s, self.feat_dim)

        # AF2-style relative position bins:
        # same-chain: [-max_relpos, ..., 0, ..., +max_relpos]
        # different-chain: one extra bin
        self.relpos_bins = 2 * self.max_relpos + 2
        self.linear_relpos = nn.Linear(self.relpos_bins, self.feat_dim)
        self.ss_embed = nn.Embedding(num_embeddings=2,embedding_dim=16)

        total_edge_feats = self.feat_dim * 3 + self._cfg.num_bins * 2 + 16 #272
        self.edge_embedder = nn.Sequential(
            nn.Linear(total_edge_feats, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.ReLU(),
            nn.Linear(self.c_p, self.c_p),
            nn.LayerNorm(self.c_p),
        )

    def embed_relpos(self, residue_index, chain_index):
        # residue_index: [B, N]
        # chain_index:     [B, N]

        same_chain = (chain_index[:, :, None] == chain_index[:, None, :])   # [B, N, N]
        rel_pos = residue_index[:, :, None] - residue_index[:, None, :]  # [B, N, N]

        rel_pos = torch.clamp(rel_pos, -self.max_relpos, self.max_relpos)
        rel_pos_bin = rel_pos + self.max_relpos  # [0, 2*max_relpos]

        diff_chain_bin = 2 * self.max_relpos + 1
        rel_pos_bin = torch.where(
            same_chain,
            rel_pos_bin,
            torch.full_like(rel_pos_bin, diff_chain_bin)
        )

        rel_pos_oh = F.one_hot(
            rel_pos_bin.long(), num_classes=self.relpos_bins
        ).float()  # [B, N, N, relpos_bins]

        return self.linear_relpos(rel_pos_oh)

    def _cross_concat(self, feats_1d, num_batch, num_res):
        return torch.cat([
            torch.tile(feats_1d[:, :, None, :], (1, 1, num_res, 1)),
            torch.tile(feats_1d[:, None, :, :], (1, num_res, 1, 1)),
        ], dim=-1).float().reshape([num_batch, num_res, num_res, -1])

    def forward(self, s, t, sc_t, p_mask, chain_index=None, residue_index=None, ss=None):
        num_batch, num_res, _ = s.shape

        if chain_index is None:
            chain_index = torch.zeros(num_batch, num_res, device=s.device, dtype=torch.long)
        if residue_index is None:
            residue_index = torch.arange(num_res, device=s.device, dtype=torch.long)[None].repeat(num_batch, 1)
        if ss is None:
            ss = torch.zeros(num_batch, num_res, num_res, device=s.device, dtype=torch.long)

        p_i = self.linear_s_p(s)                                  # [B, N, feat_dim]
        cross_node_feats = self._cross_concat(p_i, num_batch, num_res)  # [B, N, N, 2*feat_dim]

        relpos_feats = self.embed_relpos(residue_index, chain_index)       # [B, N, N, feat_dim]

        dist_feats = utils.calc_distogram(
            t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        sc_feats = utils.calc_distogram(
            sc_t, min_bin=1e-3, max_bin=20.0, num_bins=self._cfg.num_bins)
        
        ss = self.ss_embed(ss.long())

        all_edge_feats = torch.concat(
            [cross_node_feats, relpos_feats, dist_feats, sc_feats, ss], dim=-1)

        edge_feats = self.edge_embedder(all_edge_feats)
        edge_feats *= p_mask.unsqueeze(-1)
        return edge_feats
