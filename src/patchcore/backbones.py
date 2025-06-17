import timm  # noqa
import torchvision.models as models  # noqa
import onnxruntime as ortAdd commentMore actions
import tensorflow as tf

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"](pretrained="imagenet", num_classes=1000)',

    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
@@ -44,8 +45,18 @@
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
    "custom_tflite": 'load_tflite_model("models/custom_model.tflite")',
    "custom_onnx": 'load_onnx_model("models/custom_model.onnx")',
}

def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session

def load(name):
    return eval(_BACKBONES[name])
