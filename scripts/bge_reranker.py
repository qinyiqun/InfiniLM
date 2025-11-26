from typing import List, Sequence
import math
import os
from pathlib import Path
import sys
import time
import json
import torch
import transformers
from transformers import AutoModelForSequenceClassification
from collections import defaultdict

from libinfinicore_infer import (
    BGERerankerModel,
    BGERerankerMetaCStruct,
    DataType,
    DeviceType,
)
from infer_task import InferTask

from ctypes import c_float, c_int, c_uint, byref, c_uint16
import numpy as np

torch.set_default_device("cpu")

class BGERerankerMetaFromConfig(BGERerankerMetaCStruct):
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
        
        
class BGERerankerBatchedTask:
    def __init__(self, tasks: List[InferTask]):
        self.tasks = tasks
        # Precompute fields

        # Flatten token lists
        self.bsz = tasks[0].bsz
        flat_tokens = tasks[0].tokens.flatten().tolist()
        flat_masks = tasks[0].masks.flatten().tolist()
        self.ntok = int(len(flat_tokens) / self.bsz)
        
        # Convert to ctypes arrays in one pass
        self.tokens = (c_uint * (self.ntok * self.bsz))(*flat_tokens)
        elem_type = c_float if tasks[0].dtype == torch.float32 else c_uint16
        # 如果是 uint16 但数据是 float，先转整数
        flat_masks = tasks[0].masks.flatten().tolist() if tasks[0].dtype == torch.float32 else tasks[0].masks.flatten().view(torch.uint16).tolist() 
        self.masks = (elem_type * (self.ntok * self.ntok * self.bsz))(*flat_masks)  

    def input_args(self):
        return (
            self.bsz,
            self.tokens,
            self.masks,
            self.ntok,
        )
        
        
class BGERerankerForCausalLM:
    def __init__(
        self, model_dir_path, device=DeviceType.DEVICE_TYPE_CPU, ndev=1, max_tokens=None, use_fp16=False
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
        self.dtype = torch.float32 if use_fp16==False else torch.float16
        self.meta = BGERerankerMetaFromConfig(config, self.dtype, max_tokens=max_tokens) 
        
        self.bge_model = BGERerankerModel()
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
        self.load_all_safetensors_from_dir(os.path.join(model_dir_path), self.dtype)
        self.model_instance = self.bge_model.create_model(
            byref(self.meta),
            self.weights,
        )
        load_end_time = time.time()
        print(f"Time used: {load_end_time - load_start_time:.3f}s")

    def load_all_safetensors_from_dir(self, dir_path_: str, dtype:torch.dtype):
        dir_path_ = Path(dir_path_)
        model = AutoModelForSequenceClassification.from_pretrained(dir_path_, trust_remote_code=False).to(dtype)
        for name, tensor in model.state_dict().items():
            self.bge_model.load_weight(
                self.weights, name, tensor.data_ptr()
            )
            
    def max_context_len(self):
        return self.meta.dctx
    
    def batch_infer_one_round(self, tasks: InferTask):
        dense_out = (c_float * (tasks[0].bsz * 1024))() if tasks[0].dtype == torch.float32 else (c_uint16 * (tasks[0].bsz * 1024))()
        sparse_out = (c_float * (tasks[0].bsz * tasks[0].tokens.shape[1]))() if tasks[0].dtype == torch.float32 else (c_uint16 * (tasks[0].bsz * tasks[0].tokens.shape[1]))()
        batch_inputs = BGERerankerBatchedTask(tasks)
        self.bge_model.infer_batch(
            self.model_instance,
            *(batch_inputs.input_args()),
            dense_out,
            sparse_out,
        )
        return dict([('dense_vecs', dense_out),('sparse_vecs', sparse_out)])

    def generate(self, sentence_pairs, batch_size, max_length):
        assert isinstance(sentence_pairs, list)
        if isinstance(sentence_pairs[0], str):
            sentence_pairs = [sentence_pairs]
        
        # tokenize without padding to get the correct length
        all_inputs = []
        for start_index in range(0, len(sentence_pairs), batch_size):
            sentences_batch = sentence_pairs[start_index:start_index + batch_size]
            queries = [s[0] for s in sentences_batch]
            passages = [s[1] for s in sentences_batch]
            queries_inputs_batch = self.tokenizer(
                queries,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                # **kwargs
            )['input_ids']
            passages_inputs_batch = self.tokenizer(
                passages,
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
                # **kwargs
            )['input_ids']
            for q_inp, d_inp in zip(queries_inputs_batch, passages_inputs_batch):
                item = self.tokenizer.prepare_for_model(
                    q_inp,
                    d_inp,
                    truncation='only_second',
                    max_length=max_length,
                    padding=False,
                )
                all_inputs.append(item)
        # sort by length for less padding
        length_sorted_idx = np.argsort([-len(x['input_ids']) for x in all_inputs])
        all_inputs_sorted = [all_inputs[i] for i in length_sorted_idx]
        # print(all_inputs_sorted.shape)
        
        all_scores = []
        for start_index in range(0, len(all_inputs_sorted), batch_size):
            sentences_batch = all_inputs_sorted[start_index:start_index + batch_size]
            inputs = self.tokenizer.pad(
                sentences_batch,
                padding=True,
                return_tensors='pt',
            )
            
            bsz, src_len = inputs['attention_mask'].size()
            expanded_mask = inputs['attention_mask'][:, None, None, :].expand(bsz, 1, src_len, src_len).to(self.dtype)
            inverted_mask = torch.tensor(1.0, dtype=self.dtype) - expanded_mask
            inputs['attention_mask'] = inverted_mask.masked_fill(inverted_mask.to(torch.bool), torch.finfo(self.dtype).min)
            print(inputs['attention_mask'], flush=True)
            infer_task = InferTask(
                0,
                inputs['input_ids'],
                inputs['attention_mask'],
                self.max_context_len(),
                0,
                0,
                0,
                self.eos_token_id,
                self.dtype,
            )
            print(inputs)
            start_time = time.time()
            output_tokens = self.batch_infer_one_round([infer_task])
            print(output_tokens)
            
            end_time = time.time()
        return 0

            # scores = self.model(**inputs, return_dict=True).logits.view(-1, ).float()
            # all_scores.extend(scores.cpu().numpy().tolist())

        # all_scores = [all_scores[idx] for idx in np.argsort(length_sorted_idx)]

        
        # all_dense_embeddings, all_lexical_weights, all_colbert_vecs = [], [], []
        
        # def _process_token_weights(token_weights: np.ndarray, input_ids: list):
        #     # conver to dict
        #     result = defaultdict(int)
        #     unused_tokens = set()
        #     for _token in ['cls_token', 'eos_token', 'pad_token', 'unk_token']:
        #         if _token in self.tokenizer.special_tokens_map:
        #             _token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.special_tokens_map[_token])
        #             unused_tokens.add(_token_id)
        #     for w, idx in zip(token_weights, input_ids):
        #         if idx not in unused_tokens and w > 0:
        #             idx = str(idx)
        #             if w > result[idx]:
        #                 result[idx] = w
        #     return result

        # sparse_out = torch.frombuffer(output_tokens["sparse_vecs"], dtype=self.dtype, count=bsz*src_len).view(bsz, src_len, 1).clone()

        # if return_sparse:
        #     token_weights = sparse_out.squeeze(-1)
        #     all_lexical_weights.extend(
        #         list(map(
        #             _process_token_weights, 
        #             token_weights.cpu().numpy(),
        #             inputs_batch['input_ids'].cpu().numpy().tolist()
        #     )))
        # dense_out = torch.frombuffer(output_tokens["dense_vecs"], dtype=self.dtype, count=bsz*1024).view(bsz, 1024).clone()
        # if return_dense:
        #     all_dense_embeddings.append(dense_out.cpu().numpy())
            

        # if return_dense:
        #     all_dense_embeddings = np.concatenate(all_dense_embeddings, axis=0)
        #     all_dense_embeddings = all_dense_embeddings[np.argsort(length_sorted_idx)]
        #     if batch_size==1:
        #         all_dense_embeddings = all_dense_embeddings[0]

        # if return_sparse:
        #     all_lexical_weights = [all_lexical_weights[i] for i in np.argsort(length_sorted_idx)]
        #     if batch_size==1:
        #         all_lexical_weights = all_lexical_weights[0]
        # end_time = time.time()

        # return {
        #     "dense_vecs": all_dense_embeddings,
        #     "lexical_weights": all_lexical_weights,
        #     # "colbert_vecs": all_colbert_vecs 暂时不支持colber_vecs
        # }, (end_time-start_time) * 1000
    
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
    model = BGERerankerForCausalLM(model_path, device_type, ndev, use_fp16=True)
    for i in range(0,100):
        # embeddings, time = model.generate([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], 12, 8192)
        embeddings, time = model.generate([['what is panda?', 'hi'], ['what is panda?', 'The giant panda (Ailuropoda melanoleuca), sometimes called a panda bear or simply panda, is a bear species endemic to China.']], 12, 8192)
        # print(time)
        # print(embeddings)
   
    model.destroy_model_instance()


if __name__ == "__main__":
    test()