from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref, c_char_p

class BGEM3ModelWeightsCStruct(Structure):
    pass


class BGEM3ModelCStruct(Structure):
    pass

class BGEM3MetaCStruct(Structure):
    _fields_ = [
        ("dt_logits", DataType),
        ("nlayer", c_size_t),
        ("d", c_size_t),
        ("nh", c_size_t),
        ("nkvh", c_size_t),
        ("dh", c_size_t),
        ("di", c_size_t),
        ("dctx", c_size_t),
        ("dvoc", c_size_t),
        ("theta", c_float),
        ("end_token", c_uint),
    ]
    
    
@register_model
class BGEM3Model(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register BGE model functions with the library"""
        lib.createBGEM3Weights.restype = POINTER(BGEM3ModelWeightsCStruct)
        lib.createBGEM3Weights.argtypes = [
            POINTER(BGEM3MetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.createBGEM3Model.restype = POINTER(BGEM3ModelCStruct)
        lib.createBGEM3Model.argtypes = [
            POINTER(BGEM3MetaCStruct),
            POINTER(BGEM3ModelWeightsCStruct),
        ]
        
        lib.loadBGEM3ModelWeight.argtypes = [
            POINTER(BGEM3ModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]
        
    def create_model(self, meta, weights):
        return self.lib.createBGEM3Model(meta, weights)
        
    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createBGEM3Weights(meta, device_type, ndev, dev_ids)
    
    def load_weight(self, weights, name, data):
        self.lib.loadBGEM3ModelWeight(weights, name.encode("utf-8"), data)
        
    def destroy_model(self, model):
        self.lib.destroyBGEM3Model(model)
        
    def infer_batch(
        self,
        model,
        bsz,
        tokens,
        masks,
        ntok,
        # req_lens,
        # nreq,
        # req_pos,
        # kv_caches,
        # temperature,
        # topk,
        # topp,
        dense_out,
        sparse_out,
    ):
        self.lib.inferBatchBGEM3(
            model,
            bsz,
            tokens,
            masks,
            ntok,
            # req_lens,
            # nreq,
            # req_pos,
            # kv_caches,
            # temperature,
            # topk,
            # topp,
            dense_out,
            sparse_out,
        )

    def forward_batch(self, model, tokens, ntok):
        # self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    
        self.lib.forwardBatchBGEM3(
            model, tokens, ntok
            # , req_lens, nreq, req_pos, kv_caches, logits
        )