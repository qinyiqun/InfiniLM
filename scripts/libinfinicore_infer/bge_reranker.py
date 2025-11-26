from .base import BaseModel, DataType, DeviceType, KVCacheCStruct, register_model
from ctypes import c_size_t, c_uint, c_int, c_float, c_void_p, POINTER, Structure, byref, c_char_p

class BGERerankerModelWeightsCStruct(Structure):
    pass


class BGERerankerModelCStruct(Structure):
    pass

class BGERerankerMetaCStruct(Structure):
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
class BGERerankerModel(BaseModel):
    @classmethod
    def register_lib(cls, lib):
        """Register BGE model functions with the library"""
        lib.createBGERerankerWeights.restype = POINTER(BGERerankerModelWeightsCStruct)
        lib.createBGERerankerWeights.argtypes = [
            POINTER(BGERerankerMetaCStruct),
            DeviceType,
            c_int,
            POINTER(c_int),
        ]

        lib.createBGERerankerModel.restype = POINTER(BGERerankerModelCStruct)
        lib.createBGERerankerModel.argtypes = [
            POINTER(BGERerankerMetaCStruct),
            POINTER(BGERerankerModelWeightsCStruct),
        ]
        
        lib.loadBGERerankerModelWeight.argtypes = [
            POINTER(BGERerankerModelWeightsCStruct),
            c_char_p,
            c_void_p,
        ]
        
    def create_model(self, meta, weights):
        return self.lib.createBGERerankerModel(meta, weights)
        
    def create_weights(self, meta, device_type, ndev, dev_ids):
        return self.lib.createBGERerankerWeights(meta, device_type, ndev, dev_ids)
    
    def load_weight(self, weights, name, data):
        self.lib.loadBGERerankerModelWeight(weights, name.encode("utf-8"), data)
        
    def destroy_model(self, model):
        self.lib.destroyBGERerankerModel(model)
        
    def infer_batch(
        self,
        model,
        bsz,
        tokens,
        masks,
        ntok,
        dense_out,
        sparse_out,
    ):
        self.lib.inferBatchBGEReranker(
            model,
            bsz,
            tokens,
            masks,
            ntok,
            dense_out,
            sparse_out,
        )

    def forward_batch(self, model, tokens, ntok):
        self.lib.forwardBatchBGEReranker(
            model, tokens, ntok
        )