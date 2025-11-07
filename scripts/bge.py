from typing import List, Sequence
import math
import os
from pathlib import Path
import safetensors
import sys
import time
import json
import torch
import transformers

from libinfinicore_infer import (
    BGEModel,
    # JiugeAWQMetaCStruct,
    BGEMetaCStruct,
    DataType,
    DeviceType,
    KVCacheCStruct,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref

torch.set_default_device("cpu")

class BGEMetaFromConfig(BGEMetaCStruct):
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
        
        
class BGEBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        # self.kv_cache_ptrs = [t.kvcache().data() for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        flat_tokens = [tok for toks in token_lists for tok in toks]
        self.ntok = len(flat_tokens)

        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * self.ntok)(*flat_tokens)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        # self.kv_caches = (POINTER(KVCacheCStruct) * self.nreq)(*self.kv_cache_ptrs)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.tokens,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            # self.kv_caches,
            None,
            self.temperaturas,
            self.topks,
            self.topps,
        )
        
        
class BGEForCausalLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None
    ):

        load_start_time = time.time()
        print(f"Creating model on {ndev} devices...")
        with open(os.path.join(model_dir_path, "config.json"), "r") as f:
            config = json.load(f)
            self.config = config
        eos_token_id = self.config["eos_token_id"]
        self.eos_token_id = (
            [eos_token_id] if type(eos_token_id) == int else eos_token_id
        )
        self.dev_ids = (c_int * ndev)(*[i for i in range(ndev)])
        self.ndev = ndev
        self.device = device
        self.meta = BGEMetaFromConfig(config, dtype=torch.float32, max_tokens=max_tokens)

        self.bge_model = BGEModel()
        self.weights = self.bge_model.create_weights(
            byref(self.meta),
            self.device,
            ndev,
            self.dev_ids,
        )
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_dir_path, trust_remote_code=True
        )

        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

        load_start_time = time.time()
        print("Loading model weights to host...")

        self.load_all_safetensors_from_dir(os.path.join(model_dir_path))
        
        self.model_instance = self.bge_model.create_model(
            byref(self.meta),
            self.weights,
        )
        # load_end_time = time.time()
        # print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def load_all_safetensors_from_dir(self, dir_path_: str):
        dir_path_ = Path(dir_path_)
        # print(dir_path_,flush=True)
        model = transformers.AutoModel.from_pretrained(dir_path_, trust_remote_code=False)
        
        colert_linear = torch.load(dir_path_/'colbert_linear.pt', map_location='cpu', weights_only=True)
        sparse_linear = torch.load(dir_path_/'sparse_linear.pt', map_location='cpu', weights_only=True)
        
        sparse_linear['weight'] = sparse_linear['weight'].to(torch.float32)
        sparse_linear['bias'] = sparse_linear['bias'].to(torch.float32)
        
        # print('00000000000000000000000000000000000000',flush=True)
        # self.bge_model.load_weight(self.weights, 'colbert.Linear.weight', colert_linear['weight'].data_ptr())
        # print('0101010101010101010101010101',flush=True)
        
        # self.bge_model.load_weight(self.weights, 'colbert.Linear.bias', colert_linear['bias'].data_ptr())
        # print('111111111111111111111111111111111111111',flush=True)
        
        for name, tensor in model.state_dict().items():
            self.bge_model.load_weight(
                self.weights, name, tensor.data_ptr()
            )
            

        # colert_linear['weight'] =

        # print(sparse_linear['weight'],flush=True)
        # print(sparse_linear['weight'].shape, flush=True)


        self.bge_model.load_weight(self.weights, 'sparse.Linear.weight', sparse_linear['weight'].data_ptr())
        self.bge_model.load_weight(self.weights, 'sparse.Linear.bias', sparse_linear['bias'].data_ptr())
        print('22222222222222222222222222222222222',flush=True)
            
    def max_context_len(self):
        return self.meta.dctx
    
    def batch_infer_one_round(self, tasks: List[InferTask]):
        output = (c_uint * len(tasks))()
        batch_inputs = BGEBatchedTask(tasks)
        self.bge_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            output,
        )
        return list(output)

    def generate(self, input_content, max_steps, topp_=1.0, topk_=1, temperature_=1.0):
        # input_content = self.tokenizer.apply_chat_template(
        #     conversation=[{"role": "user", "content": input_content}],
        #     add_generation_prompt=True,
        #     tokenize=False,
        # )
        # print(input_content, end="", flush=True)
        tokens = self.tokenizer.encode(input_content)
        infer_task = InferTask(
            0,
            tokens,
            self.max_context_len(),
            temperature_,
            topk_,
            topp_,
            self.eos_token_id,
        )
        # infer_task.bind_kvcache(KVCache(self))

        steps = 0
        total_time = 0
        output_content = ""
        
        start_time = time.time()
        output_tokens = self.batch_infer_one_round([infer_task])
        end_time = time.time()

        # for step_i in range(max_steps):
        #     start_time = time.time()
        #     output_tokens = self.batch_infer_one_round([infer_task])
        #     end_time = time.time()
        #     steps += 1
        #     output_str = self.tokenizer.decode(output_tokens[0])
        #     output_content += output_str
        #     print(output_str, end="", flush=True)
        #     if output_tokens[0] in self.eos_token_id:
        #         break
        #     infer_task.next(output_tokens[0])

        #     if step_i > 0:
        #         total_time += end_time - start_time

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        # infer_task._kv_cache.drop(self)
        return output_content, avg_time
    
    def destroy_model_instance(self):
        self.bge_model.destroy_model(self.model_instance)
        print("Model destroyed")
            
def test():
    if len(sys.argv) < 3:
        print(
            "Usage: python jiuge_awq.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)
    model_path = sys.argv[2]
    device_type = DeviceType.DEVICE_TYPE_CPU
    if sys.argv[1] == "--cpu":
        device_type = DeviceType.DEVICE_TYPE_CPU
    elif sys.argv[1] == "--nvidia":
        device_type = DeviceType.DEVICE_TYPE_NVIDIA
    else:
        print(
            "Usage: python main_jiuge_awq.py [--cpu | --nvidia| --cambricon | --ascend | --metax | --moore] <path/to/model_dir> [n_device]"
        )
        sys.exit(1)

    ndev = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    model = BGEForCausalLM(model_path, device_type, ndev)
    model.generate("What is BGE M3?", 500)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()