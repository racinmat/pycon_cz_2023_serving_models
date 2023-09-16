import os.path as osp
import tarfile
from typing import Any
import tensorflow as tf
from ts.context import Context
from ts.torch_handler.base_handler import BaseHandler
from our_internal_package import OurCustomKerasObject


class ClassifierHandler(BaseHandler):
    def initialize(self, context: Context):
        self.manifest = context.manifest
        model_dir = context.system_properties.get("model_dir")
        tar_path = osp.join(model_dir, self.manifest["model"]["serializedFile"])
        with tarfile.open(tar_path) as tar:
            tar.extractall(model_dir)
        self.model = tf.keras.models.load_model(osp.splitext(tar_path)[0])

    def preprocess(self, requests: list[dict[str, Any]]) -> list[tf.Tensor]:
        return [tf.constant(r['body']["instances"]) for r in requests]

    def inference(self, inp_data: list[tf.Tensor]) -> list[OurCustomKerasObject]:
        # for sake of simplicity, not batching list of batches
        return [self.model.predict(i) for i in inp_data]

    def postprocess(self, output: list[OurCustomKerasObject]) -> list[
        dict[str, list[dict[str, list]]]
    ]:
        return [{"predictions":  [{
            "logits": i.numpy().tolist(),
            "confidence": i.as_confidence().numpy().tolist(),
            "prediction": [j.decode('utf-8') for j in i.as_prediction().numpy()],
        } for i in l]} for l in output]
