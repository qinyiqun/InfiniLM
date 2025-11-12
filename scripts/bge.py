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
from collections import defaultdict

from libinfinicore_infer import (
    BGEM3Model,
    JiugeAWQMetaCStruct,
    BGEM3MetaCStruct,
    DataType,
    DeviceType,
    KVCacheCStruct,
)
from infer_task import InferTask, KVCache

from ctypes import POINTER, c_float, c_int, c_uint, c_void_p, byref,cast
import numpy as np

torch.set_default_device("cpu")

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
        
        
class BGEM3BatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        self.nreq = len(tasks)

        # Precompute fields
        token_lists = [t.tokens for t in tasks]
        mask_lists = [t.masks for t in tasks]

        self.req_lens_list = [len(toks) for toks in token_lists]
        self.req_pos_list = [t.pos for t in tasks]
        self.temperaturas_list = [t.temperature for t in tasks]
        self.topks_list = [t.topk for t in tasks]
        self.topps_list = [t.topp for t in tasks]

        # Flatten token lists
        self.bsz = tasks[0].bsz
        flat_tokens = tasks[0].tokens.flatten().tolist()
        flat_masks = tasks[0].masks.flatten().tolist()
        self.ntok = int(len(flat_tokens) / self.bsz)
        
        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * (self.ntok * self.bsz))(*flat_tokens)
        self.masks = (c_float * (self.ntok * self.ntok * self.bsz))(*flat_masks)
        self.req_lens = (c_uint * self.nreq)(*self.req_lens_list)
        self.req_pos = (c_uint * self.nreq)(*self.req_pos_list)
        self.temperaturas = (c_float * self.nreq)(*self.temperaturas_list)
        self.topks = (c_uint * self.nreq)(*self.topks_list)
        self.topps = (c_float * self.nreq)(*self.topps_list)

    def input_args(self):
        return (
            self.bsz,
            self.tokens,
            self.masks,
            self.ntok,
            self.req_lens,
            self.nreq,
            self.req_pos,
            None,
            self.temperaturas,
            self.topks,
            self.topps,
        )
        
        
class BGEM3ForCausalLM:
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
        self.meta = BGEM3MetaFromConfig(config, dtype=torch.float32, max_tokens=max_tokens)

        self.bge_model = BGEM3Model()
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
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def load_all_safetensors_from_dir(self, dir_path_: str):
        dir_path_ = Path(dir_path_)
        model = transformers.AutoModel.from_pretrained(dir_path_, trust_remote_code=False)
        
        colert_linear = torch.load(dir_path_/'colbert_linear.pt', map_location='cpu', weights_only=True)
        sparse_linear = torch.load(dir_path_/'sparse_linear.pt', map_location='cpu', weights_only=True)
        
        sparse_linear['weight'] = sparse_linear['weight'].to(torch.float32)
        sparse_linear['bias'] = sparse_linear['bias'].to(torch.float32)
        
        # self.bge_model.load_weight(self.weights, 'colbert.Linear.weight', colert_linear['weight'].data_ptr())
        # print('0101010101010101010101010101',flush=True)
        
        # self.bge_model.load_weight(self.weights, 'colbert.Linear.bias', colert_linear['bias'].data_ptr())
        # print('111111111111111111111111111111111111111',flush=True)
        
        for name, tensor in model.state_dict().items():
            self.bge_model.load_weight(
                self.weights, name, tensor.data_ptr()
            )


        self.bge_model.load_weight(self.weights, 'sparse.Linear.weight', sparse_linear['weight'].data_ptr())
        self.bge_model.load_weight(self.weights, 'sparse.Linear.bias', sparse_linear['bias'].data_ptr())
            
    def max_context_len(self):
        return self.meta.dctx
    
    def batch_infer_one_round(self, tasks: InferTask):
        dense_out = (c_float * (tasks[0].bsz * 1024))()
        sparse_out = (c_float * (tasks[0].bsz * tasks[0].tokens.shape[1]))()
        batch_inputs = BGEM3BatchedTask(tasks)
        self.bge_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            dense_out,
            sparse_out,
        )
        return dict([('dense_vecs', dense_out),('sparse_vecs', sparse_out)])
        


    def generate(self, input_content, batch_size, max_length, return_dense, return_sparse):
        all_inputs = []
        for start_index in range(0, len(input_content), batch_size):
            sentences_batch = input_content[start_index:start_index + batch_size]
            inputs_batch = self.tokenizer(
                sentences_batch,
                truncation=True,
                max_length=max_length,
                # **kwargs
            )
            inputs_batch = [{
                k: inputs_batch[k][i] for k in inputs_batch.keys()
            } for i in range(len(sentences_batch))]
            all_inputs.extend(inputs_batch)

        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        
        inputs_batch = self.tokenizer.pad(
                    all_inputs_sorted[: batch_size],
                    padding=True,
                    return_tensors='pt',
                ).to('cuda')
        bsz, src_len = inputs_batch['attention_mask'].size()
        expanded_mask = inputs_batch['attention_mask'][:, None, None, :].expand(bsz, 1, src_len, src_len).to(torch.float32)
        inverted_mask = torch.tensor(1.0, dtype=torch.float32) - expanded_mask
        inputs_batch['attention_mask'] = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(torch.float32).min)

        infer_task = InferTask(
            0,
            inputs_batch['input_ids'],
            inputs_batch['attention_mask'],
            self.max_context_len(),
            0,
            0,
            0,
            self.eos_token_id,
        )

        steps = 0
        total_time = 0
        
        start_time = time.time()
        output_tokens = self.batch_infer_one_round([infer_task])
        
        all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []
        
        def _process_token_weights(token_weights: np.ndarray, input_ids: list):
            # conver to dict
            result = defaultdict(int)
            unused_tokens = set()
            for _token in ['cls_token', 'eos_token', 'pad_token', 'unk_token']:
                if _token in self.tokenizer.special_tokens_map:
                    _token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map[_token])
                    unused_tokens.add(_token_id)
            for w, idx in zip(token_weights, input_ids):
                if idx not in unused_tokens and w > 0:
                    idx = str(idx)
                    if w > result[idx]:
                        result[idx] = w
            return result

        sparse_out = torch.frombuffer(output_tokens["sparse_vecs"], dtype=torch.float32, count=bsz*src_len).view(bsz, src_len, 1).clone()

        if return_sparse:
            token_weights = sparse_out.squeeze(-1)
            all_lexical_weights.extend(
                list(map(
                    _process_token_weights, 
                    token_weights.cpu().numpy(),
                    inputs_batch['input_ids'].cpu().numpy().tolist()
            )))
        dense_out = torch.frombuffer(output_tokens["dense_vecs"], dtype=torch.float32, count=bsz*1024).view(bsz, 1024).clone()
        if return_dense:
            all_dense_embeddings.append(dense_out.cpu().numpy())
            

        if return_dense:
            all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
            all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]
            if batch_size==1:
                all_dense_embeddings = all_dense_embeddings[0]

        if return_sparse:
            all_lexical_weights = [all_lexical_weights[i] for i in np.argsort(length_sorted_idx)]
            if batch_size==1:
                all_lexical_weights = all_lexical_weights[0]
        end_time = time.time()

        print("\n")
        avg_time = total_time * 1000 / (steps - 1)
        print(f"Time per step: {avg_time:.3f}ms")

        return {
            "dense_vecs": all_dense_embeddings,
            "lexical_weights": all_lexical_weights,
            "colbert_vecs": all_colbert_vecs
        }, (end_time-start_time) * 1000
    
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
    model = BGEM3ForCausalLM(model_path, device_type, ndev)
    for i in range(0,1000):
        embeddings, time = model.generate(["What is BGE M3?", "What the fuck you are doing, you hit me, you a damn guy", "Tell me, look in my eyes, tell me.", "EVE is the shabiest game in china, its planner is the shabiest people."], 12, 8192, True, True)
        print(time)
    # print(embeddings['dense_vecs'])
    # print(embeddings['lexical_weights'])
    # model.generate(["What the fuck you are doing, you hit me, you a damn guy", "What the fuck you are doing, you hit me, you a damn guy"], 12, 8192)
    model.destroy_model_instance()


if __name__ == "__main__":
    test()