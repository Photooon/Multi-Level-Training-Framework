import torch.nn as nn

class MatmulMod:
    """
    Do matrix multiplication for module.
    """
    @classmethod
    def param(cls, w, B=None, A=None, conv=False):
        # TODO Reconstruct this part
        out = w
        if B != None:
            if conv:
                out = B.matmul(out.view(out.size(0), -1))
                out = out.reshape(out.size(0), w.size(1), w.size(2), w.size(3))
            else:
                out = B.matmul(out)
        if A != None:
            # TODO: Make it more clear
            if conv:
                out = out.transpose(0, 1)
                shape = out.shape
                out = out.reshape(out.size(0), -1)
                out = (A.T.matmul(out))
                out = out.reshape(out.size(0), shape[1], shape[2], shape[3])
                out = out.transpose(0, 1)
            else:
                out = out.matmul(A)
        return out
    
    @classmethod
    def linear_in_place(cls, mod, B, A):
        cls.linear(mod, mod, B, A)

    @classmethod
    def linear(cls, src_mod, tgt_mod, B, A):
        tgt_mod.weight.data = cls.param(src_mod.weight.data, B, A, conv=isinstance(src_mod, nn.Conv2d))
        if hasattr(tgt_mod, "bias") and tgt_mod.bias is not None:
            tgt_mod.bias.data = cls.param(src_mod.bias.data, B, None)

    @classmethod
    def layernorm_in_place(cls, mod, B):
        cls.layernorm(mod, mod, B)
 
    @classmethod
    def layernorm(cls, src_mod, tgt_mod, B):
        tgt_mod.weight.data = cls.param(src_mod.weight.data, B)
        tgt_mod.bias.data = cls.param(src_mod.bias.data, B)


class MatmulLayers:
    """
    Do depth matrix multiplication for layers
    """
    @classmethod
    def layers(cls, src_layers, tgt_layers, M):
        assert M.size(0) == len(tgt_layers)
        assert M.size(1) == len(src_layers)

        for t, tl in enumerate(tgt_layers):
            weighted_params = []
            for s, sl in enumerate(src_layers):
                param_len = len(list(sl.parameters()))
                for p, param in enumerate(list(sl.parameters())):
                    if len(weighted_params) < param_len:
                        weighted_params.append(param.data.clone() * M[t, s])
                    else:
                        weighted_params[p] += param.data.clone() * M[t, s]
            for p, param in enumerate(list(tl.parameters())):
                param.data[:] = weighted_params[p]
        