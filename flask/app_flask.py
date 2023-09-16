from flask import Flask, request, jsonify
import tensorflow as tf
import os.path as osp
from our_internal_package import OurCustomModel

app = Flask(__name__)
model: OurCustomModel = tf.keras.models.load_model(osp.join(osp.dirname(__file__),
                                                        'our_trained_model'))


@app.route('/predict', methods=['POST'])
def predict():
    logits = model.predict(tf.constant([request.get_data()]))
    response = {
        "logits": logits.numpy().tolist(),
        "confidence": logits.as_confidence().numpy().tolist(),
        "prediction": [j.decode('utf-8') for j in logits.as_prediction().numpy()],
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run()
