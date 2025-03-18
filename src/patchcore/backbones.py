import timm  # noqa
import torchvision.models as models  # noqa
import onnxruntime as ort
import tensorflow as tf
import ctypes
import requests

_BACKBONES = {
    "alexnet": "models.alexnet(pretrained=True)",
    "bninception": 'pretrainedmodels.__dict__["bninception"](pretrained="imagenet", num_classes=1000)',
    "resnet50": "models.resnet50(pretrained=True)",
    "resnet101": "models.resnet101(pretrained=True)",
    "resnext101": "models.resnext101_32x8d(pretrained=True)",
    "resnet200": 'timm.create_model("resnet200", pretrained=True)',
    "resnest50": 'timm.create_model("resnest50d_4s2x40d", pretrained=True)',
    "resnetv2_50_bit": 'timm.create_model("resnetv2_50x3_bitm", pretrained=True)',
    "resnetv2_50_21k": 'timm.create_model("resnetv2_50x3_bitm_in21k", pretrained=True)',
    "resnetv2_101_bit": 'timm.create_model("resnetv2_101x3_bitm", pretrained=True)',
    "resnetv2_101_21k": 'timm.create_model("resnetv2_101x3_bitm_in21k", pretrained=True)',
    "resnetv2_152_bit": 'timm.create_model("resnetv2_152x4_bitm", pretrained=True)',
    "resnetv2_152_21k": 'timm.create_model("resnetv2_152x4_bitm_in21k", pretrained=True)',
    "resnetv2_152_384": 'timm.create_model("resnetv2_152x2_bit_teacher_384", pretrained=True)',
    "resnetv2_101": 'timm.create_model("resnetv2_101", pretrained=True)',
    "vgg11": "models.vgg11(pretrained=True)",
    "vgg19": "models.vgg19(pretrained=True)",
    "vgg19_bn": "models.vgg19_bn(pretrained=True)",
    "wideresnet50": "models.wide_resnet50_2(pretrained=True)",
    "wideresnet101": "models.wide_resnet101_2(pretrained=True)",
    "mnasnet_100": 'timm.create_model("mnasnet_100", pretrained=True)',
    "mnasnet_a1": 'timm.create_model("mnasnet_a1", pretrained=True)',
    "mnasnet_b1": 'timm.create_model("mnasnet_b1", pretrained=True)',
    "densenet121": 'timm.create_model("densenet121", pretrained=True)',
    "densenet201": 'timm.create_model("densenet201", pretrained=True)',
    "inception_v4": 'timm.create_model("inception_v4", pretrained=True)',
    "vit_small": 'timm.create_model("vit_small_patch16_224", pretrained=True)',
    "vit_base": 'timm.create_model("vit_base_patch16_224", pretrained=True)',
    "vit_large": 'timm.create_model("vit_large_patch16_224", pretrained=True)',
    "vit_r50": 'timm.create_model("vit_large_r50_s32_224", pretrained=True)',
    "vit_deit_base": 'timm.create_model("deit_base_patch16_224", pretrained=True)',
    "vit_deit_distilled": 'timm.create_model("deit_base_distilled_patch16_224", pretrained=True)',
    "vit_swin_base": 'timm.create_model("swin_base_patch4_window7_224", pretrained=True)',
    "vit_swin_large": 'timm.create_model("swin_large_patch4_window7_224", pretrained=True)',
    "efficientnet_b7": 'timm.create_model("tf_efficientnet_b7", pretrained=True)',
    "efficientnet_b5": 'timm.create_model("tf_efficientnet_b5", pretrained=True)',
    "efficientnet_b3": 'timm.create_model("tf_efficientnet_b3", pretrained=True)',
    "efficientnet_b1": 'timm.create_model("tf_efficientnet_b1", pretrained=True)',
    "efficientnetv2_m": 'timm.create_model("tf_efficientnetv2_m", pretrained=True)',
    "efficientnetv2_l": 'timm.create_model("tf_efficientnetv2_l", pretrained=True)',
    "efficientnet_b3a": 'timm.create_model("efficientnet_b3a", pretrained=True)',
    "tflite": 'download_and_load_model("https://huggingface.co/qualcomm/WideResNet50-Quantized/resolve/main/WideResNet50-Quantized.so","custom_model.so")',
    "onnx": 'download_and_load_model("https://huggingface.co/qualcomm/WideResNet50-Quantized/resolve/main/WideResNet50-Quantized.so","custom_model.so")',
    "so": 'load_so_model("https://huggingface.co/qualcomm/WideResNet50-Quantized/resolve/main/WideResNet50-Quantized.so","custom_model.so")',
}

def load_so_model(url, model_filename):
    """Downloads a model file from a URL and loads it using ctypes.

    Args:
        url: The URL of the model file to download.
        model_filename: The filename to save the model as.

    Returns:
        The loaded model object.
    """
    # Download the model file
    response = requests.get(url, stream=True)
    with open(model_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    # Load the model using ctypes
    model = ctypes.CDLL(model_filename)
    return model


def load_tflite_model(model_path):
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def load_onnx_model(model_path):
    session = ort.InferenceSession(model_path)
    return session



def load(name):
    return eval(_BACKBONES[name])
