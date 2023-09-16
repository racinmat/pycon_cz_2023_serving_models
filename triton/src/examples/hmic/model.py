from typing import Optional
import triton_python_backend_utils as pb_utils
import sys
from wanna_model.model import WannaModel
import tensorflow as tf
import os.path as osp


class TritonPythonModel:
    def __init__(self):
        self.model: Optional[WannaModel] = None
        print(f"Python version:{sys.version}")
        print(f"Version info:{sys.version_info}")

    @staticmethod
    def auto_complete_config(auto_complete_model_config):
        # some disgusting configuration
        inputs = [{
            'name': 'INPUT0',
            'data_type': 'TYPE_STRING',
            'dims': [1]
        }]
        # the dims of output must correspond to the number of predicted classes, which can change in case of strain and
        # type models
        outputs = [{
            'name': 'OUTPUT0',
            'data_type': 'TYPE_FP32',
            'dims': [2]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_FP32',
            'dims': [2]
        }, {
            'name': 'OUTPUT1',
            'data_type': 'TYPE_STRING',
            'dims': [1]
        }]

        # Demonstrate the usage of `as_dict`, `add_input`, `add_output`,
        # `set_max_batch_size`, and `set_dynamic_batching` functions.
        # Store the model configuration as a dictionary.
        config = auto_complete_model_config.as_dict()
        input_names = []
        output_names = []
        for input in config['input']:
            input_names.append(input['name'])
        for output in config['output']:
            output_names.append(output['name'])

        for input in inputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_input` will check for conflicts and
            # raise errors if an input with the same name already exists in
            # the configuration but has different data_type or dims property.
            if input['name'] not in input_names:
                auto_complete_model_config.add_input(input)
        for output in outputs:
            # The name checking here is only for demonstrating the usage of
            # `as_dict` function. `add_output` will check for conflicts and
            # raise errors if an output with the same name already exists in
            # the configuration but has different data_type or dims property.
            if output['name'] not in output_names:
                auto_complete_model_config.add_output(output)

        # we don't know what is max batch size reasonable for hmic, so this is a wild guess
        auto_complete_model_config.set_max_batch_size(20)

        # To enable a dynamic batcher with default settings, you can use
        # auto_complete_model_config set_dynamic_batching() function. It is
        # commented in this example because the max_batch_size is zero.
        #
        # auto_complete_model_config.set_dynamic_batching()

        return auto_complete_model_config

    def initialize(self, args):
        model_path = osp.join(args['model_repository'], args['name'], args['version'])
        self.model = tf.keras.models.load_model(model_path, 'out_custom_keras_objects')

    def execute(self, requests):
        responses = []
        inputs = []
        for request in requests:
            print(f'{request=}')
            in0 = pb_utils.get_input_tensor_by_name(request, "INPUT0")
            inputs.append(in0)
        print(f'{inputs=}')
        outputs = self.model(tf.constant(inputs))
        # somehow turn outputs to responses, I did not manage to make it actually run
        return responses

    def finalize(self):
        print('Cleaning up...')
