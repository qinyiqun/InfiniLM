import torch
import transformers
import os
import json
from libinfinicore_infer import DataType, DeviceType, BGEM3MetaCStruct

class BGEM3MetaFromConfig(BGEM3MetaCStruct):
    def __init__(self, config, dtype=torch.float16, max_tokens=None):
        if dtype == torch.float16:
            dt_ = DataType.INFINI_DTYPE_F16
        elif dtype == torch.float32:
            dt_ = DataType.INFINI_DTYPE_F32
        elif dtype == torch.bfloat16:
            dt_ = DataType.INFINI_DTYPE_BF16
        else:
            dt_ = DataType.INFINI_DTYPE_F16
 
        super().__init__(
            dt_logits=dt_,
            nlayer=config["num_hidden_layers"],
            d=config["hidden_size"],
            nh=config["num_attention_heads"],
            nkvh=(
                config["num_key_value_heads"]
                if "num_key_value_heads" in config
                else config["num_attention_heads"]
            ),
            dh=(
                config["head_dim"]
                if "head_dim" in config
                else config["hidden_size"] // config["num_attention_heads"]
            ),
            di=config["intermediate_size"],
            dctx=(
                config["max_position_embeddings"] if max_tokens is None else max_tokens
            ),
            dvoc=config["vocab_size"],
            end_token=2,
        )
        self.torch_dtype_logits = dtype
        
        


model_dir_or_path = "/data/shared/qinyiqun/bge_dev/bge-m3"


with open(os.path.join(model_dir_or_path, "config.json"), "r") as f:
    config = json.load(f)
    # print(config)

meta = BGEM3MetaFromConfig(config, dtype=torch.float16, max_tokens=8192)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_dir_or_path, trust_remote_code=True)
model = transformers.AutoModel.from_pretrained(model_dir_or_path, trust_remote_code=False)
model = model.eval().to("cuda").half()

for name, tensor in model.state_dict().items():
    if name == "embeddings.word_embeddings.weight":
        print(f"Loaded weight: {name}, shape: {tensor.shape}, dtype: {tensor.dtype}, ptr: {tensor.data_ptr()}")

colbert_model_path = os.path.join(model_dir_or_path, 'colbert_linear.pt')
sparse_model_path = os.path.join(model_dir_or_path, 'sparse_linear.pt')

colbert_state_dict = torch.load(colbert_model_path, map_location='cpu', weights_only=True)
sparse_state_dict = torch.load(sparse_model_path, map_location='cpu', weights_only=True)

print(colbert_state_dict)
print(colbert_state_dict['weight'].data_ptr())

# for name, tensor in model.state_dict().items():
#     print(f"{name:60s}  {tensor.shape}")
# print(model.state_dict().keys())

# print(colbert_state_dict)
print(sparse_state_dict)
print(sparse_state_dict['weight'].shape)
print(sparse_state_dict['weight'])
print(sparse_state_dict['bias'].shape)
print(sparse_state_dict['bias'])