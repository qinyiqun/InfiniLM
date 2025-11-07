from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref, c_char_p

class ModelWeightsCStruct(Structure):
    pass


class BGEModelCStruct(Structure):
    pass

class BGEMetaCStruct(Structure):
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
class BGEModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register BGE model functions with the library"""
        lib.createBGEWeights.restype = POINTER(ModelWeightsCStruct)
        lib.createBGEWeights.argtypes = [
            POINTER(BGEMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.createBGEModel.restype = POINTER(BGEModelCStruct)
        lib.createBGEModel.argtypes = [
            POINTER(BGEMetaCStruct),
            POINTER(ModelWeightsCStruct),
        ]
        
        lib.loadModelWeight.argtypes = [
            POINTER(ModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]
        
    def create_model(self, meta, weights):
        return self.lib.createBGEModel(meta, weights)
        
    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createBGEWeights(meta, device_type, ndev, dev_ids)
    
    def load_weight(self, weights, name, data):
        self.lib.loadModelWeight(weights, name.encode("utf-8"), data)
        
    def destroy_model(self, model):
        self.lib.destroyBGEModel(model)
        
    def infer_batch(
        self,
        model,
        tokens,
        ntok,
        req_lens,
        nreq,
        req_pos,
        kv_caches,
        temperature,
        topk,
        topp,
        output,
    ):
        self.lib.inferBatchBGE(
            model,
            tokens,
            ntok,
            req_lens,
            nreq,
            req_pos,
            kv_caches,
            temperature,
            topk,
            topp,
            output,
        )

    def forward_batch(
        self, model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
    ):
        self.lib.forwardBatchBGE(
            model, tokens, ntok, req_lens, nreq, req_pos, kv_caches, logits
        )