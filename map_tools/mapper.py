import os
import json
import torch
import torch.nn as nn

from tools import MatmulMod, MatmulLayers
from argparse import ArgumentParser


class MapMatrix:
    @classmethod
    def norm(cls, M, dim=0):
        # dim=0: col norm
        # dim=1: row norm
        M_sum = M.sum(dim=dim)
        if dim == 0:
            M_sum = M_sum.view(1, -1)
        else:
            M_sum = M_sum.view(-1, 1)
        M_sum[M_sum == 0] = 1
        if torch.sum(M_sum == 0) != 0:
            print("Warning: some rows/cols sum in map matrix are zeros, these rows/cols would keep zeros after normalization.")
        return M / M_sum

    @classmethod
    def get_F(cls, src_width, tgt_width, scheme=None, head_size=64):
        delta_width = src_width - tgt_width
        assert delta_width >= 0

        if scheme is not None:
            assert len(scheme) == tgt_width // head_size
            embed_F = torch.zeros((tgt_width, src_width))
            qk_F = torch.zeros((tgt_width, src_width))
            v_F = torch.zeros((tgt_width, src_width))
            inter_F = torch.zeros((tgt_width * 4, src_width * 4))

            def set_chunk_eye(F, i, j, head_size, times=1):
                chunk_size = head_size * times
                F[i*chunk_size:(i+1)*chunk_size, j*chunk_size:(j+1)*chunk_size] = torch.eye(chunk_size)

            for i, sub_scheme in enumerate(scheme):
                for j in sub_scheme:
                    set_chunk_eye(embed_F, i, j, head_size=head_size)
                    set_chunk_eye(qk_F, i, j, head_size=head_size)
                    set_chunk_eye(v_F, i, j, head_size=head_size)
                    set_chunk_eye(inter_F, i, j, head_size=head_size, times=4)

            embed_F = cls.norm(embed_F, dim=1)
            qk_F = cls.norm(qk_F, dim=1)
            v_F = cls.norm(v_F, dim=1)
            inter_F = cls.norm(inter_F, dim=1)
        else:   # Default Template
            embed_F = cls.norm(torch.cat([torch.eye(tgt_width), torch.eye(tgt_width, delta_width)], dim=1), dim=1)
            qk_F = cls.norm(torch.cat([torch.eye(tgt_width), torch.eye(tgt_width, delta_width)], dim=1), dim=1)
            v_F = cls.norm(torch.cat([torch.eye(tgt_width), torch.eye(tgt_width, delta_width)], dim=1), dim=1)
            inter_F = cls.norm(torch.cat([torch.eye(tgt_width * 4), torch.eye(tgt_width * 4, delta_width * 4)], dim=1), dim=1)
        
        return embed_F, qk_F, v_F, inter_F

    @classmethod
    def get_T(cls, embed_F, qk_F, v_F, inter_F):
        embed_T = cls.norm(embed_F.T, dim=1)
        qk_T = cls.norm(qk_F.T, dim=1)
        v_T = cls.norm(v_F.T, dim=1)
        inter_T = cls.norm(inter_F.T, dim=1)
        
        return embed_T, qk_T, v_T, inter_T

    @classmethod
    def get_R(cls, src_depth, tgt_depth, scheme=None):
        if src_depth == tgt_depth:
            return torch.eye(src_depth)
        
        R = torch.zeros((tgt_depth, src_depth))

        if scheme is not None:
            assert len(scheme) == tgt_depth
            for i, sub_scheme in enumerate(scheme):
                for j in sub_scheme:
                    R[i, j] = 1 / len(sub_scheme)
        else:   # Default: Merge adjacent layers
            for i in range(tgt_depth):
                R[i, 2 * i] = 0.5
                R[i, 2 * i + 1] = 0.5
        
        return R

    @classmethod
    def get_G(cls, R):
        return cls.norm(R.T, dim=1)


class TransformerMapper:
    def __init__(self, src_dir: str, tgt_dir: str, save_dir: str, 
                 width_scheme: list=None, depth_scheme: list=None, load_tgt_state: bool=False):
        self.src_dir = src_dir
        self.tgt_dir = tgt_dir
        self.save_dir = save_dir
        self.width_scheme = width_scheme
        self.depth_scheme = depth_scheme

        # Build Models
        self.src_state = torch.load(os.path.join(src_dir, "pytorch_model.bin"), map_location="cpu")
        if load_tgt_state:
            self.tgt_state = torch.load(os.path.join(tgt_dir, "pytorch_model.bin"), map_location="cpu")
        else:
            self.tgt_state = None
        self.src_model = self._build_model(self.src_state, src_dir)
        self.tgt_model = self._build_model(self.tgt_state, tgt_dir)

        self.src_width, self.src_depth = self._get_model_attr(self.src_model)
        self.tgt_width, self.tgt_depth = self._get_model_attr(self.tgt_model)

    def _build_model(self, state, dir_path: str):
        raise NotImplementedError()

    def _get_model_attr(self, model):
        raise NotImplementedError()
    
    def _build_mapmatrix(self):
        map_matrix_path = os.path.join(self.src_dir, "MapMatrix.bin")
        if os.path.exists(map_matrix_path):
            self.map_matrix = torch.load(map_matrix_path)
        else:
            # if in decoalesce mode, we should reverse the src and tgt to get F and R
            sw, tw = max(self.src_width, self.tgt_width), min(self.src_width, self.tgt_width)
            sd, td = max(self.src_depth, self.tgt_depth), min(self.src_depth, self.tgt_depth)
            embed_F, qk_F, v_F, inter_F = MapMatrix.get_F(sw, tw, scheme=self.width_scheme)
            self.map_matrix = {
                "embed_F": embed_F,
                "qk_F": qk_F,
                "v_F": v_F,
                "inter_F": inter_F,
                "R": MapMatrix.get_R(sd, td, scheme=self.depth_scheme)
            }

    def coalesce(self):
        self._build_mapmatrix()
        if self.src_width != self.tgt_width:
            self.coalesce_width(self.map_matrix["embed_F"], self.map_matrix["qk_F"], 
                                self.map_matrix["v_F"], self.map_matrix["inter_F"])
        self.coalesce_depth(self.map_matrix["R"])
        return self

    def coalesce_width(self, embed_F, qk_F, v_F, inter_F):
        raise NotImplementedError()
    
    def coalesce_depth(self, R):
        raise NotImplementedError()

    def decoalesce(self):
        self._build_mapmatrix()
        if self.src_width != self.tgt_width:
            embed_T, qk_T, v_T, inter_T = MapMatrix.get_T(self.map_matrix["embed_F"], self.map_matrix["qk_F"], 
                                                          self.map_matrix["v_F"], self.map_matrix["inter_F"])
            self.decoalesce_width(embed_T, qk_T, v_T, inter_T)
        G = MapMatrix.get_G(self.map_matrix["R"])
        self.decoalesce_depth(G)
        return self

    def decoalesce_width(self, embed_T, qk_T, v_T, inter_T):
        raise NotImplementedError()

    def decoalesce_depth(self, G):
        raise NotImplementedError()
    
    def interpolate(self, alpha: float=0.25, **kwargs):
        with torch.no_grad():
            for (n, src_param), (_, tgt_param) in zip(list(self.src_model.named_parameters()), list(self.tgt_model.named_parameters())):
                # Inject
                tgt_param.data = (1 - alpha) * tgt_param.data + alpha * src_param.data

        return self
    
    def matmul_model(self):
        raise NotImplementedError()

    def save(self):
        if self.save_dir is not None:
            # Save Model
            tgt_state = self.tgt_model.state_dict()
            torch.save(tgt_state, os.path.join(self.save_dir, "pytorch_model.bin"))
            
            # Save MapMatrix
            map_matrix_path = os.path.join(self.src_dir, "MapMatrix.bin")
            if not os.path.exists(map_matrix_path) and getattr(self, "map_matrix", None) is not None:
                torch.save(self.map_matrix, os.path.join(self.save_dir, "MapMatrix.bin"))
                print("Model and MapMatrix Saved.")
        else:
            raise ValueError("Save Path is None.")

class BertMapper(TransformerMapper):
    def _build_model(self, state, dir_path: str):
        from transformers.models.bert import BertForMaskedLM
        from transformers.models.bert import BertConfig

        config = BertConfig.from_json_file(os.path.join(dir_path, "config.json"))
        model = BertForMaskedLM(config)
        if state is not None:
            model.load_state_dict(state, strict=False)

        return model

    def _get_model_attr(self, model):
        return model.bert.embeddings.word_embeddings.weight.size(1), len(model.bert.encoder.layer)

    def coalesce_width(self, embed_F, qk_F, v_F, inter_F):
        # In-place Coalescing
        with torch.no_grad():
            self.matmul_model(embed_F, qk_F, v_F, inter_F)
            self.src_model.bert.embeddings.word_embeddings.weight *= 2
            self.src_model.bert.embeddings.position_embeddings.weight *= 2
            self.src_model.bert.embeddings.token_type_embeddings.weight *= 2

    def coalesce_depth(self, R):
        with torch.no_grad():
            self.tgt_model.bert.embeddings.word_embeddings.weight.data = self.src_model.bert.embeddings.word_embeddings.weight.data
            self.tgt_model.bert.embeddings.position_embeddings.weight.data = self.src_model.bert.embeddings.position_embeddings.weight.data
            self.tgt_model.bert.embeddings.token_type_embeddings.weight.data = self.src_model.bert.embeddings.token_type_embeddings.weight.data
            self.tgt_model.bert.embeddings.LayerNorm.weight.data = self.src_model.bert.embeddings.LayerNorm.weight.data
            self.tgt_model.bert.embeddings.LayerNorm.bias.data = self.src_model.bert.embeddings.LayerNorm.bias.data

            self.tgt_model.cls.predictions.transform.dense.weight.data = self.src_model.cls.predictions.transform.dense.weight.data
            self.tgt_model.cls.predictions.transform.dense.bias.data = self.src_model.cls.predictions.transform.dense.bias.data
            self.tgt_model.cls.predictions.transform.LayerNorm.weight.data = self.src_model.cls.predictions.transform.LayerNorm.weight.data
            self.tgt_model.cls.predictions.transform.LayerNorm.bias.data = self.src_model.cls.predictions.transform.LayerNorm.bias.data
            self.tgt_model.cls.predictions.decoder.bias.data = self.src_model.cls.predictions.decoder.bias.data

            MatmulLayers.layers(self.src_model.bert.encoder.layer, self.tgt_model.bert.encoder.layer, M=R)

    def decoalesce_width(self, embed_T, qk_T, v_T, inter_T):
        # In-place Decoalescing
        with torch.no_grad():
            self.matmul_model(embed_T, qk_T, v_T, inter_T)
            self.src_model.bert.embeddings.word_embeddings.weight *= 0.5
            self.src_model.bert.embeddings.position_embeddings.weight *= 0.5
            self.src_model.bert.embeddings.token_type_embeddings.weight *= 0.5

    def decoalesce_depth(self, G):
        with torch.no_grad():
            self.tgt_model.bert.embeddings.word_embeddings.weight.data = self.src_model.bert.embeddings.word_embeddings.weight.data
            self.tgt_model.bert.embeddings.position_embeddings.weight.data = self.src_model.bert.embeddings.position_embeddings.weight.data
            self.tgt_model.bert.embeddings.token_type_embeddings.weight.data = self.src_model.bert.embeddings.token_type_embeddings.weight.data
            self.tgt_model.bert.embeddings.LayerNorm.weight.data = self.src_model.bert.embeddings.LayerNorm.weight.data
            self.tgt_model.bert.embeddings.LayerNorm.bias.data = self.src_model.bert.embeddings.LayerNorm.bias.data

            self.tgt_model.cls.predictions.transform.dense.weight.data = self.src_model.cls.predictions.transform.dense.weight.data
            self.tgt_model.cls.predictions.transform.dense.bias.data = self.src_model.cls.predictions.transform.dense.bias.data
            self.tgt_model.cls.predictions.transform.LayerNorm.weight.data = self.src_model.cls.predictions.transform.LayerNorm.weight.data
            self.tgt_model.cls.predictions.transform.LayerNorm.bias.data = self.src_model.cls.predictions.transform.LayerNorm.bias.data
            self.tgt_model.cls.predictions.decoder.bias.data = self.src_model.cls.predictions.decoder.bias.data

            MatmulLayers.layers(self.src_model.bert.encoder.layer, self.tgt_model.bert.encoder.layer, M=G)

    def matmul_model(self, embed_M, qk_M, v_M, inter_M):
        # In-place width adjustment
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.word_embeddings, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.position_embeddings, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.token_type_embeddings, B=None, A=embed_M.T)
        MatmulMod.layernorm_in_place(self.src_model.bert.embeddings.LayerNorm, B=embed_M)

        for src_layer in self.src_model.bert.encoder.layer:
            MatmulMod.linear_in_place(src_layer.attention.self.query, B=qk_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.self.key, B=qk_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.self.value, B=v_M, A=MapMatrix.norm(qk_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.output.dense, B=embed_M, A=MapMatrix.norm(v_M.T, dim=1))
            MatmulMod.layernorm_in_place(src_layer.attention.output.LayerNorm, embed_M)
            MatmulMod.linear_in_place(src_layer.intermediate.dense, B=inter_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.output.dense, B=embed_M, A=MapMatrix.norm(inter_M.T, dim=1))
            MatmulMod.layernorm_in_place(src_layer.output.LayerNorm, B=embed_M)

        MatmulMod.linear_in_place(self.src_model.cls.predictions.transform.dense, B=embed_M, A=MapMatrix.norm(embed_M.T, dim=1))
        MatmulMod.layernorm_in_place(self.src_model.cls.predictions.transform.LayerNorm, B=embed_M)

class BertPreLNMapper(BertMapper):
    def _build_model(self, state, dir_path: str):
        from models.bert_preln import BertPreLNForMaskedLM
        from models.configuration_bert_preln import BertPreLNConfig

        config = BertPreLNConfig.from_json_file(os.path.join(dir_path, "config.json"))
        model = BertPreLNForMaskedLM(config)
        if state is not None:
            model.load_state_dict(state, strict=False)

        return model
    
    def coalesce_depth(self, R):
        with torch.no_grad():
            self.tgt_model.bert.embeddings.word_embeddings.weight.data = self.src_model.bert.embeddings.word_embeddings.weight.data
            self.tgt_model.bert.embeddings.position_embeddings.weight.data = self.src_model.bert.embeddings.position_embeddings.weight.data
            self.tgt_model.bert.embeddings.token_type_embeddings.weight.data = self.src_model.bert.embeddings.token_type_embeddings.weight.data

            self.tgt_model.cls.predictions.transform.dense.weight.data = self.src_model.cls.predictions.transform.dense.weight.data
            self.tgt_model.cls.predictions.transform.dense.bias.data = self.src_model.cls.predictions.transform.dense.bias.data
            self.tgt_model.cls.predictions.transform.LayerNorm.weight.data = self.src_model.cls.predictions.transform.LayerNorm.weight.data
            self.tgt_model.cls.predictions.transform.LayerNorm.bias.data = self.src_model.cls.predictions.transform.LayerNorm.bias.data
            self.tgt_model.cls.predictions.decoder.bias.data = self.src_model.cls.predictions.decoder.bias.data

            MatmulLayers.layers(self.src_model.bert.encoder.layer, self.tgt_model.bert.encoder.layer, M=R)

    def decoalesce_depth(self, G):
        with torch.no_grad():
            self.tgt_model.bert.embeddings.word_embeddings.weight.data = self.src_model.bert.embeddings.word_embeddings.weight.data
            self.tgt_model.bert.embeddings.position_embeddings.weight.data = self.src_model.bert.embeddings.position_embeddings.weight.data
            self.tgt_model.bert.embeddings.token_type_embeddings.weight.data = self.src_model.bert.embeddings.token_type_embeddings.weight.data

            self.tgt_model.cls.predictions.transform.dense.weight.data = self.src_model.cls.predictions.transform.dense.weight.data
            self.tgt_model.cls.predictions.transform.dense.bias.data = self.src_model.cls.predictions.transform.dense.bias.data
            self.tgt_model.cls.predictions.transform.LayerNorm.weight.data = self.src_model.cls.predictions.transform.LayerNorm.weight.data
            self.tgt_model.cls.predictions.transform.LayerNorm.bias.data = self.src_model.cls.predictions.transform.LayerNorm.bias.data
            self.tgt_model.cls.predictions.decoder.bias.data = self.src_model.cls.predictions.decoder.bias.data

            MatmulLayers.layers(self.src_model.bert.encoder.layer, self.tgt_model.bert.encoder.layer, M=G)

    def matmul_model(self, embed_M, qk_M, v_M, inter_M):
        # In-place width adjustment
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.word_embeddings, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.position_embeddings, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.token_type_embeddings, B=None, A=embed_M.T)

        for src_layer in self.src_model.bert.encoder.layer:
            MatmulMod.layernorm_in_place(src_layer.LayerNorm_1, embed_M)
            MatmulMod.linear_in_place(src_layer.attention.self.query, B=qk_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.self.key, B=qk_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.self.value, B=v_M, A=MapMatrix.norm(qk_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.output.dense, B=embed_M, A=MapMatrix.norm(v_M.T, dim=1))
            MatmulMod.layernorm_in_place(src_layer.LayerNorm_2, embed_M)
            MatmulMod.linear_in_place(src_layer.intermediate.dense, B=inter_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.output.dense, B=embed_M, A=MapMatrix.norm(inter_M.T, dim=1))

        MatmulMod.linear_in_place(self.src_model.cls.predictions.transform.dense, B=embed_M, A=MapMatrix.norm(embed_M.T, dim=1))
        MatmulMod.layernorm_in_place(self.src_model.cls.predictions.transform.LayerNorm, B=embed_M)

class BertPreLN2Mapper(BertMapper):
    def _build_model(self, state, dir_path: str):
        from models.bert_preln2 import BertPreLN2ForMaskedLM
        from models.configuration_bert_preln2 import BertPreLN2Config

        config = BertPreLN2Config.from_json_file(os.path.join(dir_path, "config.json"))
        model = BertPreLN2ForMaskedLM(config)
        if state is not None:
            model.load_state_dict(state, strict=False)

        return model
    
    def coalesce_depth(self, R):
        with torch.no_grad():
            self.tgt_model.bert.embeddings.word_embeddings.weight.data = self.src_model.bert.embeddings.word_embeddings.weight.data
            self.tgt_model.bert.embeddings.position_embeddings.weight.data = self.src_model.bert.embeddings.position_embeddings.weight.data
            self.tgt_model.bert.embeddings.token_type_embeddings.weight.data = self.src_model.bert.embeddings.token_type_embeddings.weight.data

            self.tgt_model.cls.predictions.LayerNorm.weight.data = self.src_model.cls.predictions.LayerNorm.weight.data
            self.tgt_model.cls.predictions.LayerNorm.bias.data = self.src_model.cls.predictions.LayerNorm.bias.data
            self.tgt_model.cls.predictions.decoder.bias.data = self.src_model.cls.predictions.decoder.bias.data

            MatmulLayers.layers(self.src_model.bert.encoder.layer, self.tgt_model.bert.encoder.layer, M=R)

    def decoalesce_depth(self, G):
        with torch.no_grad():
            self.tgt_model.bert.embeddings.word_embeddings.weight.data = self.src_model.bert.embeddings.word_embeddings.weight.data
            self.tgt_model.bert.embeddings.position_embeddings.weight.data = self.src_model.bert.embeddings.position_embeddings.weight.data
            self.tgt_model.bert.embeddings.token_type_embeddings.weight.data = self.src_model.bert.embeddings.token_type_embeddings.weight.data

            self.tgt_model.cls.predictions.LayerNorm.weight.data = self.src_model.cls.predictions.LayerNorm.weight.data
            self.tgt_model.cls.predictions.LayerNorm.bias.data = self.src_model.cls.predictions.LayerNorm.bias.data
            self.tgt_model.cls.predictions.decoder.bias.data = self.src_model.cls.predictions.decoder.bias.data

            MatmulLayers.layers(self.src_model.bert.encoder.layer, self.tgt_model.bert.encoder.layer, M=G)

    def matmul_model(self, embed_M, qk_M, v_M, inter_M):
        # In-place width adjustment
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.word_embeddings, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.position_embeddings, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.bert.embeddings.token_type_embeddings, B=None, A=embed_M.T)

        for src_layer in self.src_model.bert.encoder.layer:
            MatmulMod.layernorm_in_place(src_layer.LayerNorm_1, embed_M)
            MatmulMod.linear_in_place(src_layer.attention.self.query, B=qk_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.self.key, B=qk_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.self.value, B=v_M, A=MapMatrix.norm(qk_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.attention.output.dense, B=embed_M, A=MapMatrix.norm(v_M.T, dim=1))
            MatmulMod.layernorm_in_place(src_layer.LayerNorm_2, embed_M)
            MatmulMod.linear_in_place(src_layer.intermediate.dense, B=inter_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_layer.output.dense, B=embed_M, A=MapMatrix.norm(inter_M.T, dim=1))

        MatmulMod.layernorm_in_place(self.src_model.cls.predictions.LayerNorm, B=embed_M)

class GPT2Mapper(TransformerMapper):
    def _build_model(self, state, dir_path: str):
        from transformers.models.gpt2 import GPT2LMHeadModel
        from transformers.models.gpt2 import GPT2Config

        config = GPT2Config.from_json_file(os.path.join(dir_path, "config.json"))
        model = GPT2LMHeadModel(config)
        if state is not None:
            model.load_state_dict(state, strict=False)

        return model
    
    def _get_model_attr(self, model):
        return model.transformer.wte.weight.size(1), len(model.transformer.h)

    def coalesce_width(self, embed_F, qk_F, v_F, inter_F):
        # In-place Coalescing
        with torch.no_grad():
            self.matmul_model(embed_F, qk_F, v_F, inter_F)
            self.src_model.transformer.wte.weight *= 2
            self.src_model.transformer.wpe.weight *= 2

    def coalesce_depth(self, R):
        with torch.no_grad():
            self.tgt_model.transformer.wte.weight.data = self.src_model.transformer.wte.weight.data
            self.tgt_model.transformer.wpe.weight.data = self.src_model.transformer.wpe.weight.data
            self.tgt_model.transformer.ln_f.weight.data = self.src_model.transformer.ln_f.weight.data
            self.tgt_model.transformer.ln_f.bias.data = self.src_model.transformer.ln_f.bias.data

            MatmulLayers.layers(self.src_model.transformer.h, self.tgt_model.transformer.h, M=R)

    def decoalesce_width(self, embed_T, qk_T, v_T, inter_T):
        # In-place Decoalescing
        with torch.no_grad():
            self.matmul_model(embed_T, qk_T, v_T, inter_T)
            self.src_model.transformer.wte.weight *= 0.5
            self.src_model.transformer.wpe.weight *= 0.5

    def decoalesce_depth(self, G):
        with torch.no_grad():
            self.tgt_model.transformer.wte.weight.data = self.src_model.transformer.wte.weight.data
            self.tgt_model.transformer.wpe.weight.data = self.src_model.transformer.wpe.weight.data
            self.tgt_model.transformer.ln_f.weight.data = self.src_model.transformer.ln_f.weight.data
            self.tgt_model.transformer.ln_f.bias.data = self.src_model.transformer.ln_f.bias.data

            MatmulLayers.layers(self.src_model.transformer.h, self.tgt_model.transformer.h, M=G)
    
    def matmul_model(self, embed_M, qk_M, v_M, inter_M):
        # In-place width adjustment
        MatmulMod.linear_in_place(self.src_model.transformer.wte, B=None, A=embed_M.T)
        MatmulMod.linear_in_place(self.src_model.transformer.wpe, B=None, A=embed_M.T)

        for src_h in self.src_model.transformer.h:
            src_h.attn.c_attn.weight.transpose_(1, 0)
            src_h.attn.c_proj.weight.transpose_(1, 0)
            src_h.mlp.c_fc.weight.transpose_(1, 0)
            src_h.mlp.c_proj.weight.transpose_(1, 0)

            MatmulMod.layernorm_in_place(src_h.ln_1, B=embed_M)

            src_qkv_w = list(src_h.attn.c_attn.weight.chunk(3, dim=0))
            src_qkv_b = list(src_h.attn.c_attn.bias.chunk(3, dim=0))
            qkv_M = [qk_M, qk_M, v_M]
            for i in range(3):
                B_w = qkv_M[i]
                A_w = embed_M if i != 2 else qk_M
                A_w = MapMatrix.norm(A_w.T, dim=1)
                src_qkv_w[i].data = MatmulMod.param(src_qkv_w[i], B=B_w, A=A_w)
                src_qkv_b[i].data = MatmulMod.param(src_qkv_b[i], B=qkv_M[i], A=None)
            src_h.attn.c_attn.weight.data = torch.cat(src_qkv_w, dim=0)
            src_h.attn.c_attn.bias.data = torch.cat(src_qkv_b, dim=0)

            MatmulMod.linear_in_place(src_h.attn.c_proj, B=embed_M, A=MapMatrix.norm(v_M.T, dim=1))
            MatmulMod.layernorm_in_place(src_h.ln_2, B=embed_M)
            MatmulMod.linear_in_place(src_h.mlp.c_fc, B=inter_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_h.mlp.c_proj, B=embed_M, A=MapMatrix.norm(inter_M.T, dim=1))
            
            src_h.attn.c_attn.weight.transpose_(1, 0)
            src_h.attn.c_proj.weight.transpose_(1, 0)
            src_h.mlp.c_fc.weight.transpose_(1, 0)
            src_h.mlp.c_proj.weight.transpose_(1, 0)

        MatmulMod.layernorm_in_place(self.src_model.transformer.ln_f, B=embed_M)

class DeiTMapper(TransformerMapper):
    def __init__(self, src_model_type: str, tgt_model_type: str,
                 src_dir: str, src_ckp_name: str, 
                 tgt_dir: str=None, tgt_ckp_name: str=None,
                 save_dir: str=None, save_ckp_name: str=None,
                 width_scheme: list=None, depth_scheme: list=None):
        self.src_model_type = src_model_type
        self.tgt_model_type = tgt_model_type
        self.src_dir = src_dir
        self.save_dir = save_dir
        self.save_ckp_name = save_ckp_name
        self.width_scheme = width_scheme
        self.depth_scheme = depth_scheme

        # Build Models
        self.src_state = torch.load(os.path.join(src_dir, src_ckp_name), map_location="cpu")
        if tgt_dir is not None and tgt_ckp_name is not None:
            self.tgt_state = torch.load(os.path.join(tgt_dir, tgt_ckp_name), map_location="cpu")
        else:
            self.tgt_state = None
        self.src_model = self._build_model(self.src_state, src_model_type)
        self.tgt_model = self._build_model(self.tgt_state, tgt_model_type)

        self.src_width, self.src_depth = self._get_model_attr(self.src_model)
        self.tgt_width, self.tgt_depth = self._get_model_attr(self.tgt_model)

    def _build_model(self, state, model_type: str):
        from functools import partial
        from timm.models.vision_transformer import VisionTransformer, _cfg

        arch_dict = {
            "L6-H3": (6, 3),    # depth, num_heads
            "L12-H3": (12, 3),
            "L6-H6": (6, 6),
            "L12-H6": (12, 6),
            "L12-H3": (12, 3),
            "L12-H12": (12, 12)
        }

        model = VisionTransformer(
            patch_size=16, embed_dim=arch_dict[model_type][1] * 64, depth=arch_dict[model_type][0], num_heads=arch_dict[model_type][1], mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()

        if state is not None:
            if "model" in state:
                state = state["model"]
            model.load_state_dict(state)

        return model
    
    def _get_model_attr(self, model):
        return model.patch_embed.proj.weight.size(0), len(model.blocks)

    def coalesce_width(self, embed_F, qk_F, v_F, inter_F):
        # In-place Coalescing
        with torch.no_grad():
            self.matmul_model(embed_F, qk_F, v_F, inter_F)
    
    def coalesce_depth(self, R):
        with torch.no_grad():
            self.tgt_model.cls_token.data = self.src_model.cls_token
            self.tgt_model.pos_embed.data = self.src_model.pos_embed.data
            self.tgt_model.patch_embed.proj.weight.data = self.src_model.patch_embed.proj.weight.data
            self.tgt_model.patch_embed.proj.bias.data = self.src_model.patch_embed.proj.bias.data 

            self.tgt_model.norm.weight.data = self.src_model.norm.weight.data
            self.tgt_model.norm.bias.data = self.src_model.norm.bias.data
            self.tgt_model.head.weight.data = self.src_model.head.weight.data
            self.tgt_model.head.bias.data = self.src_model.head.bias.data
    
            MatmulLayers.layers(self.src_model.blocks, self.tgt_model.blocks, M=R)

    def decoalesce_width(self, embed_T, qk_T, v_T, inter_T):
        # In-place Decoalescing
        with torch.no_grad():
            self.matmul_model(embed_T, qk_T, v_T, inter_T)

    def decoalesce_depth(self, G):
        with torch.no_grad():
            self.tgt_model.cls_token.data = self.src_model.cls_token
            self.tgt_model.pos_embed.data = self.src_model.pos_embed.data
            self.tgt_model.patch_embed.proj.weight.data = self.src_model.patch_embed.proj.weight.data
            self.tgt_model.patch_embed.proj.bias.data = self.src_model.patch_embed.proj.bias.data 

            self.tgt_model.norm.weight.data = self.src_model.norm.weight.data
            self.tgt_model.norm.bias.data = self.src_model.norm.bias.data
            self.tgt_model.head.weight.data = self.src_model.head.weight.data
            self.tgt_model.head.bias.data = self.src_model.head.bias.data
    
            MatmulLayers.layers(self.src_model.blocks, self.tgt_model.blocks, M=G)

    def matmul_model(self, embed_M, qk_M, v_M, inter_M):
        # In-place width adjustment

        # cls_token
        self.src_model.cls_token.data = MatmulMod.param(self.src_model.cls_token.data[0][0], B=embed_M).reshape(1, 1, embed_M.size(0))

        # patch_embed
        MatmulMod.linear_in_place(self.src_model.patch_embed.proj, B=embed_M, A=None)

        # pos_embed
        self.src_model.pos_embed.data = MatmulMod.param(self.src_model.pos_embed.data[0], B=None, A=embed_M.T).reshape(1, self.src_model.pos_embed.size(1), embed_M.size(0))

        for src_block in self.src_model.blocks:
            MatmulMod.layernorm_in_place(src_block.norm1, B=embed_M)
            # attention
            src_qkv_w = list(src_block.attn.qkv.weight.chunk(3, dim=0))
            src_qkv_b = list(src_block.attn.qkv.bias.chunk(3, dim=0))

            qkv_M = [qk_M, qk_M, v_M]
            for i in range(3):
                B_w = qkv_M[i]
                A_w = embed_M if i != 2 else qk_M
                A_w = MapMatrix.norm(A_w.T, dim=1)
                src_qkv_w[i].data = MatmulMod.param(src_qkv_w[i], B=B_w, A=A_w)
                src_qkv_b[i].data = MatmulMod.param(src_qkv_b[i], B=qkv_M[i], A=None)
            src_block.attn.qkv.weight.data = torch.cat(src_qkv_w, dim=0)
            src_block.attn.qkv.bias.data = torch.cat(src_qkv_b, dim=0)

            MatmulMod.linear_in_place(src_block.attn.proj, B=embed_M, A=MapMatrix.norm(embed_M.T, dim=1))

            MatmulMod.layernorm_in_place(src_block.norm2, B=embed_M)
            MatmulMod.linear_in_place(src_block.mlp.fc1, B=inter_M, A=MapMatrix.norm(embed_M.T, dim=1))
            MatmulMod.linear_in_place(src_block.mlp.fc2, B=embed_M, A=MapMatrix.norm(inter_M.T, dim=1))

        # norm, head
        MatmulMod.layernorm_in_place(self.src_model.norm, B=embed_M)
        MatmulMod.linear_in_place(self.src_model.head, B=None, A=MapMatrix.norm(embed_M.T, dim=1))

    def save(self):
        if self.save_dir is not None:
            # Save Model
            tgt_state = self.tgt_model.state_dict()
            torch.save(tgt_state, os.path.join(self.save_dir, self.save_ckp_name))
            
            # Save MapMatrix
            map_matrix_path = os.path.join(self.src_dir, "MapMatrix.bin")
            if not os.path.exists(map_matrix_path):
                self.map_matrix = torch.save(self.map_matrix, os.path.join(self.save_dir, "MapMatrix.bin"))

            print("Model and MapMatrix Saved.")
        else:
            raise ValueError("Save Path is None.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--arch", type=str, default="bert", help="Architecture")
    parser.add_argument("--src-arch", type=str, default=None, help="Source Model Architecture")
    parser.add_argument("--tgt-arch", type=str, default=None, help="Target Model Architecture")
    parser.add_argument("--src-dir", type=str, default=None, help="Source Model Dir Path")
    parser.add_argument("--tgt-dir", type=str, default=None, help="Target Model Dir Path")
    parser.add_argument("--src-ckp", type=str, default=None, help="Source Model Checkpoint Name")
    parser.add_argument("--tgt-ckp", type=str, default=None, help="Target Model Checkpoint Name")
    parser.add_argument("--opt", type=str, default="C", help="Select Operator")
    parser.add_argument("--config", type=str, default=None, required=True, help="Mapping Configuration")
    parser.add_argument("--alpha", type=float, default=0, help="Interpolation Ratio")
    parser.add_argument("--save-dir", type=str, default=None, help="Save Dir Path")
    parser.add_argument("--save-ckp", type=str, default=None, help="Save Model Checkpoint Name")
    args = parser.parse_args()

    with open(args.config, 'r') as mf:
        config = json.load(mf)

    if args.arch == "bert":
        mapper = BertMapper(src_dir=args.src_dir, tgt_dir=args.tgt_dir, save_dir=args.save_dir, 
                            width_scheme=config.get("width_scheme", None), depth_scheme=config.get("depth_scheme", None), 
                            load_tgt_state=args.opt=="IP")
    elif args.arch == "bert_preln":
        mapper = BertPreLNMapper(src_dir=args.src_dir, tgt_dir=args.tgt_dir, save_dir=args.save_dir, 
                                 width_scheme=config.get("width_scheme", None), depth_scheme=config.get("depth_scheme", None), 
                                 load_tgt_state=args.opt=="IP")
    elif args.arch == "bert_preln2":
        mapper = BertPreLN2Mapper(src_dir=args.src_dir, tgt_dir=args.tgt_dir, save_dir=args.save_dir, 
                                 width_scheme=config.get("width_scheme", None), depth_scheme=config.get("depth_scheme", None), 
                                 load_tgt_state=args.opt=="IP")
    elif args.arch == "gpt2":
        mapper = GPT2Mapper(src_dir=args.src_dir, tgt_dir=args.tgt_dir, save_dir=args.save_dir, 
                            width_scheme=config.get("width_scheme", None), depth_scheme=config.get("depth_scheme", None), 
                            load_tgt_state=args.opt=="IP")
    elif args.arch == "deit":
        mapper = DeiTMapper(src_model_type=args.src_arch, tgt_model_type=args.tgt_arch, 
                            src_dir=args.src_dir, src_ckp_name=args.src_ckp, 
                            tgt_dir=args.tgt_dir, tgt_ckp_name=args.tgt_ckp, 
                            save_dir=args.save_dir, save_ckp_name=args.save_ckp, 
                            width_scheme=config.get("width_scheme", None), depth_scheme=config.get("depth_scheme", None))
    else:
        raise ValueError(f"Architecture: {args.arch} is Undefined.")

    if args.opt == "C":
        mapper.coalesce()
    elif args.opt == "D":
        mapper.decoalesce()
    elif args.opt == "IP":
        mapper.interpolate(args.alpha, reserved_heads=config.get("reserved_heads", None), reserved_layers=config.get("reserved_layers", None))
    else:
        raise ValueError(f"Operator: {args.opt} is Undefined.")

    mapper.save()

    print(f"{args.opt} Finished.")