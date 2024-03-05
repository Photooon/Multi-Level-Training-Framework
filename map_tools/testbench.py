# Test for different models mapping
import os
import json
import torch

from mapper import BertMapper, GPT2Mapper, DeiTMapper

def load_json(path):
    with open(path, 'r') as mf:
        config = json.load(mf)
    return config

def diff_transformer(src_model, tgt_model, input_sz, randint=True):
    with torch.no_grad():
        src_model.eval()
        tgt_model.eval()

        if randint:
            tokens = torch.randint(0, 30522, input_sz).long()
            src_out = src_model(tokens).logits
            tgt_out = tgt_model(tokens).logits
        else:
            imgs = torch.randn(input_sz)
            src_out = src_model(imgs)        
            tgt_out = tgt_model(imgs)        

        delta = (src_out - tgt_out).abs().mean()
        return delta

def verify_bert():
    # L6-E384 -> L6-E768 -> L6-E384: diff nearly equals to 1e-6
    # L6-E384 -> L12-E384 -> L6-E384: diff nearly equals to 0
    input_sz = (2, 512)

    def build_bert(dir_path, load_state=True):
        from transformers.models.bert import BertForMaskedLM
        from transformers.models.bert import BertConfig
        config = BertConfig.from_json_file(os.path.join(dir_path, "config.json"))
        model = BertForMaskedLM(config)
        if load_state:
            state = torch.load(os.path.join(dir_path, "pytorch_model.bin"), map_location=torch.device('cpu'))
            model.load_state_dict(state)
        return model

    base_model = build_bert("test_models/bert-L6-E384", load_state=False)
    torch.save(base_model.state_dict(), "test_models/bert-L6-E384/pytorch_model.bin")

    config = load_json("config/L6-H6.json")
    
    print(f"-------------BERT Coalesce & Decoalesce Verify-------------")
    print(f"Config: Width Scheme={config['width_scheme']}, Depth Scheme={config['depth_scheme']}")
    mapper1 = BertMapper(src_dir="test_models/bert-L6-E384", tgt_dir="test_models/bert-L6-E768", 
                        save_dir="test_models/bert-L6-E768", width_scheme=config["width_scheme"], depth_scheme=None)
    mapper1.decoalesce().save()

    mapper2 = BertMapper(src_dir="test_models/bert-L6-E768", tgt_dir="test_models/bert-L6-E384", 
                        save_dir=None, width_scheme=config["width_scheme"], depth_scheme=None)
    mapper2.coalesce()

    print(f"L6-E384 -> L6-E768: {diff_transformer(base_model, mapper1.tgt_model, input_sz): .1e}")
    print(f"L6-E768 -> L6-E384: {diff_transformer(mapper1.tgt_model, mapper2.tgt_model, input_sz): .1e}")

    mapper3 = BertMapper(src_dir="test_models/bert-L6-E384", tgt_dir="test_models/bert-L12-E384",
                        save_dir="test_models/bert-L12-E384", width_scheme=None, depth_scheme=config["depth_scheme"])
    mapper3.decoalesce().save()

    mapper4 = BertMapper(src_dir="test_models/bert-L12-E384", tgt_dir="test_models/bert-L6-E384",
                        save_dir=None, width_scheme=None, depth_scheme=config["depth_scheme"])
    mapper4.coalesce()

    print(f"L6-E384 -> L12-E384: {diff_transformer(base_model, mapper3.tgt_model, input_sz): .1e}")
    print(f"L6-E384 -> L12-E384 -> L6-E384: {diff_transformer(base_model, mapper4.tgt_model, input_sz): .1e}")
    print(f"-----------BERT Coalesce & Decoalesce Verify End-----------")

    print(f"--------BERT Interpolation Verify--------")
    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        mapper5 = BertMapper(src_dir="test_models/bert-L6-E384", tgt_dir="test_models/bert-L6-E384", 
                             save_dir=None, load_tgt_state=True)
        mapper5.src_model = build_bert("test_models/bert-L6-E384", load_state=False)    # inject an initialized model to a well trained model
        mapper5.interpolate(alpha=alpha)
        print(f"Alpha={alpha}: {diff_transformer(base_model, mapper5.tgt_model, input_sz): .1e}")
    print(f"------BERT Interpolation Verify End------")

def verify_gpt2():
    # L6-E384 -> L6-E768 -> L6-E384: diff nearly equals to 7e-1
    # L6-E384 -> L12-E384 -> L6-E384: diff nearly equals to 0
    input_sz = (2, 512)

    def build_gpt2(dir_path, load_state=True):
        from transformers.models.gpt2 import GPT2LMHeadModel
        from transformers.models.gpt2 import GPT2Config
        config = GPT2Config.from_json_file(os.path.join(dir_path, "config.json"))
        model = GPT2LMHeadModel(config)
        if load_state:
            state = torch.load(os.path.join(dir_path, "pytorch_model.bin"), map_location=torch.device('cpu'))
            model.load_state_dict(state)
        return model

    base_model = build_gpt2("test_models/gpt2-L6-E384", load_state=False)
    torch.save(base_model.state_dict(), "test_models/gpt2-L6-E384/pytorch_model.bin")

    config = load_json("config/L6-H6.json")

    print(f"-------------GPT2 Coalesce & Decoalesce Verify-------------")
    print(f"Config: Width Scheme={config['width_scheme']}, Depth Mode={config['depth_scheme']}")
    mapper1 = GPT2Mapper(src_dir="test_models/gpt2-L6-E384", tgt_dir="test_models/gpt2-L6-E768", 
                            save_dir="test_models/gpt2-L6-E768", width_scheme=config["width_scheme"], depth_scheme=None)
    mapper1.decoalesce().save()

    mapper2 = GPT2Mapper(src_dir="test_models/gpt2-L6-E768", tgt_dir="test_models/gpt2-L6-E384", 
                            save_dir=None, width_scheme=config["width_scheme"], depth_scheme=None)
    mapper2.coalesce()

    print(f"L6-E384 -> L6-E768: {diff_transformer(base_model, mapper1.tgt_model, input_sz): .1e}")
    print(f"L6-E768 -> L6-E384: {diff_transformer(mapper1.tgt_model, mapper2.tgt_model, input_sz): .1e}")

    mapper3 = GPT2Mapper(src_dir="test_models/gpt2-L6-E384", tgt_dir="test_models/gpt2-L12-E384",
                            save_dir="test_models/gpt2-L12-E384", width_scheme=None, depth_scheme=config["depth_scheme"])
    mapper3.decoalesce().save()

    mapper4 = GPT2Mapper(src_dir="test_models/gpt2-L12-E384", tgt_dir="test_models/gpt2-L6-E384",
                            save_dir=None, width_scheme=None, depth_scheme=config["depth_scheme"])
    mapper4.coalesce()

    print(f"L6-E384 -> L12-E384: {diff_transformer(base_model, mapper3.tgt_model, input_sz): .1e}")
    print(f"L6-E384 -> L12-E384 -> L6-E384: {diff_transformer(base_model, mapper4.tgt_model, input_sz): .1e}")
    print(f"-----------GPT2 Verify End-----------")

    print(f"--------GPT2 Interpolation Verify--------")
    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        mapper5 = GPT2Mapper(src_dir="test_models/gpt2-L6-E384", tgt_dir="test_models/gpt2-L6-E384", 
                             save_dir=None, load_tgt_state=True)
        mapper5.src_model = build_gpt2("test_models/gpt2-L6-E384", load_state=False)    # inject an initialized model to a well trained model
        mapper5.interpolate(alpha=alpha)
        print(f"Alpha={alpha}: {diff_transformer(base_model, mapper5.tgt_model, input_sz): .1e}")
    print(f"------GPT2 Interpolation Verify End------")

def verify_deit():
    # L6-H6 -> L6-H12 -> L6-H6: diff nearly equals to 7e-6
    # L6-H6 -> L12-H6 -> L6-H6: diff nearly equals to 0
    input_sz = (1, 3, 224, 224)

    def build_deit(model_type: str, checkpoint_path=None):
        from functools import partial
        from timm.models.vision_transformer import VisionTransformer, _cfg

        arch_dict = {
            "L6-H3": (6, 3),    # depth, num_heads
            "L6-H6": (6, 6),
            "L6-H12": (6, 12),
            "L12-H3": (12, 3),
            "L12-H12": (12, 12)
        }

        model = VisionTransformer(
            patch_size=16, embed_dim=arch_dict[model_type][1] * 64, depth=arch_dict[model_type][0], num_heads=arch_dict[model_type][1], mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6))
        model.default_cfg = _cfg()

        if checkpoint_path is not None:
            state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
            if state is not None:
                if "model" in state:
                    state = state["model"]
                model.load_state_dict(state)

        return model

    base_model = build_deit("L6-H3", "test_models/deit-L6-H3/checkpoint-130.pth")

    config = load_json("config/L6-H6.json")
    print(f"-------------DeiT Coalesce & Decoalesce Verify-------------")
    print(f"Config: Width Scheme={config['width_scheme']}, Depth Scheme={config['depth_scheme']}")
    mapper1 = DeiTMapper(src_model_type="L6-H3", tgt_model_type="L6-H6", src_dir="test_models/deit-L6-H3", src_ckp_name="checkpoint-130.pth",
                            save_dir="test_models/deit-L6-H6", save_ckp_name="checkpoint-DW.pth", width_scheme=config["width_scheme"], depth_scheme=None)
    mapper1.decoalesce().save()

    mapper2 = DeiTMapper(src_model_type="L6-H6", tgt_model_type="L6-H3", src_dir="test_models/deit-L6-H6", src_ckp_name="checkpoint-DW.pth",
                            save_dir=None, save_ckp_name=None, width_scheme=config["width_scheme"], depth_scheme=None)
    mapper2.coalesce()

    print(f"L6-H3 -> L6-H6: {diff_transformer(base_model, mapper1.tgt_model, input_sz, randint=False): .1e}")
    print(f"L6-H6 -> L6-H3: {diff_transformer(mapper1.tgt_model, mapper2.tgt_model, input_sz, randint=False): .1e}")

    mapper3 = DeiTMapper(src_model_type="L6-H3", tgt_model_type="L12-H3", src_dir="test_models/deit-L6-H3", src_ckp_name="checkpoint-130.pth",
                            save_dir="test_models/deit-L12-H3", save_ckp_name="checkpoint-DD.pth", width_scheme=None, depth_scheme=config["depth_scheme"])
    mapper3.decoalesce().save()

    mapper4 = DeiTMapper(src_model_type="L12-H3", tgt_model_type="L6-H3", src_dir="test_models/deit-L12-H3", src_ckp_name="checkpoint-DD.pth",
                            save_dir=None, save_ckp_name=None, width_scheme=None, depth_scheme=config["depth_scheme"])
    mapper4.coalesce()

    print(f"L6-H3 -> L12-H3: {diff_transformer(base_model, mapper3.tgt_model, input_sz, randint=False): .1e}")
    print(f"L6-H3 -> L12-H3 -> L6-H3: {diff_transformer(base_model, mapper4.tgt_model, input_sz, randint=False): .1e}")
    print(f"-----------DeiT Coalesce & Decoalesce Verify End-----------")

    print(f"--------DeiT Interpolation Verify--------")
    for alpha in [0, 0.25, 0.5, 0.75, 1]:
        mapper5 = DeiTMapper(src_model_type="L6-H3", tgt_model_type="L6-H3", 
                             src_dir="test_models/deit-L6-H3", src_ckp_name="checkpoint-130.pth",
                             tgt_dir="test_models/deit-L6-H3", tgt_ckp_name="checkpoint-130.pth",
                             save_dir=None, save_ckp_name=None)
        mapper5.src_model = build_deit("L6-H3")    # inject an initialized model to a well trained model
        mapper5.interpolate(alpha=alpha)
        print(f"Alpha={alpha}: {diff_transformer(base_model, mapper5.tgt_model, input_sz, randint=False): .1e}")
    print(f"------DeiT Interpolation Verify End------")

if __name__ == "__main__":
    verify_bert()
    verify_gpt2()
    # verify_deit()