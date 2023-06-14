import onnx_graphsurgeon as gs
from onnx.external_data_helper import convert_model_to_external_data
import onnx
import os

MODEL_PATH = "."
SIZE_THRESHOLD = 1024
EXPORT_DIR = "."


def export_external(name: str, model: gs.Graph):
    """Export an ONNX model to external data format."""
    model.cleanup().toposort()
    gs_export = gs.export_onnx(model, do_type_check=True)
    convert_model_to_external_data(
        gs_export, all_tensors_to_one_file=False, size_threshold=SIZE_THRESHOLD
    )
    if not os.path.exists(EXPORT_DIR):
        os.makedirs(EXPORT_DIR)
    onnx.save(gs_export, "{}/{}".format(EXPORT_DIR, name))


encoder_model = gs.import_onnx(onnx.load("{}/{}".format(MODEL_PATH, "t5-large_encoder_decoder_init.onnx")))
decoder_model = gs.import_onnx(onnx.load("{}/{}".format(MODEL_PATH, "t5-large_decoder.onnx")))

export_external("enc_dec_init.onnx", encoder_model)
export_external("dec.onnx", decoder_model)



