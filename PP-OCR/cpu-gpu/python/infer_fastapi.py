# Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import fastdeploy as fd
import cv2
import os
import uvicorn
from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from pydantic import BaseModel, Field
import re
from urllib.request import urlopen
import numpy as np
from pathlib import Path


class Config:
    det_model: str      # Path of Detection model of PPOCR.
    cls_model: str      # Path of Classification model of PPOCR.
    rec_model: str      # Path of Recognization model of PPOCR.
    rec_label_file: str # Path of Recognization label of PPOCR.
    device: int         # Type of inference device, support 'cpu' or 'gpu'.
    device_id: int      # Define which GPU card used to run model.
    cls_bs: int         # Classification model inference batch size.
    rec_bs: int         # Recognition model inference batch size
    backend: str        # Type of inference backend, support ort/trt/paddle/openvino, default 'openvino' for cpu, 'tensorrt' for gpu


def parse_arguments():
    config = Config()
    config.det_model = r"ch_PP-OCRv3_det_infer"
    config.cls_model = r"ch_ppocr_mobile_v2.0_cls_infer"
    config.rec_model = r"ch_PP-OCRv3_rec_infer"
    config.rec_label_file = r"ppocr_keys_v1.txt"
    config.device = r"cpu"
    config.device_id = 0
    config.cls_bs = 1
    config.rec_bs = 6
    config.backend = "default"
    return config


def build_option(args):

    det_option = fd.RuntimeOption()
    cls_option = fd.RuntimeOption()
    rec_option = fd.RuntimeOption()

    if args.device.lower() == "gpu":
        det_option.use_gpu(args.device_id)
        cls_option.use_gpu(args.device_id)
        rec_option.use_gpu(args.device_id)

    if args.backend.lower() == "trt":
        assert args.device.lower(
        ) == "gpu", "TensorRT backend require inference on device GPU."
        det_option.use_trt_backend()
        cls_option.use_trt_backend()
        rec_option.use_trt_backend()

        # If use TRT backend, the dynamic shape will be set as follow.
        # We recommend that users set the length and height of the detection model to a multiple of 32.
        # We also recommend that users set the Trt input shape as follow.
        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                       [1, 3, 960, 960])
        cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.cls_bs, 3, 48, 320],
                                       [args.cls_bs, 3, 48, 1024])
        rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.rec_bs, 3, 48, 320],
                                       [args.rec_bs, 3, 48, 2304])

        # Users could save TRT cache file to disk as follow.
        det_option.set_trt_cache_file(args.det_model + "/det_trt_cache.trt")
        cls_option.set_trt_cache_file(args.cls_model + "/cls_trt_cache.trt")
        rec_option.set_trt_cache_file(args.rec_model + "/rec_trt_cache.trt")

    elif args.backend.lower() == "pptrt":
        assert args.device.lower(
        ) == "gpu", "Paddle-TensorRT backend require inference on device GPU."
        det_option.use_paddle_infer_backend()
        det_option.paddle_infer_option.collect_trt_shape = True
        det_option.paddle_infer_option.enable_trt = True

        cls_option.use_paddle_infer_backend()
        cls_option.paddle_infer_option.collect_trt_shape = True
        cls_option.paddle_infer_option.enable_trt = True

        rec_option.use_paddle_infer_backend()
        rec_option.paddle_infer_option.collect_trt_shape = True
        rec_option.paddle_infer_option.enable_trt = True

        # If use TRT backend, the dynamic shape will be set as follow.
        # We recommend that users set the length and height of the detection model to a multiple of 32.
        # We also recommend that users set the Trt input shape as follow.
        det_option.set_trt_input_shape("x", [1, 3, 64, 64], [1, 3, 640, 640],
                                       [1, 3, 960, 960])
        cls_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.cls_bs, 3, 48, 320],
                                       [args.cls_bs, 3, 48, 1024])
        rec_option.set_trt_input_shape("x", [1, 3, 48, 10],
                                       [args.rec_bs, 3, 48, 320],
                                       [args.rec_bs, 3, 48, 2304])

        # Users could save TRT cache file to disk as follow.
        det_option.set_trt_cache_file(args.det_model)
        cls_option.set_trt_cache_file(args.cls_model)
        rec_option.set_trt_cache_file(args.rec_model)

    elif args.backend.lower() == "ort":
        det_option.use_ort_backend()
        cls_option.use_ort_backend()
        rec_option.use_ort_backend()

    elif args.backend.lower() == "paddle":
        det_option.use_paddle_infer_backend()
        cls_option.use_paddle_infer_backend()
        rec_option.use_paddle_infer_backend()

    elif args.backend.lower() == "openvino":
        assert args.device.lower(
        ) == "cpu", "OpenVINO backend require inference on device CPU."
        det_option.use_openvino_backend()
        cls_option.use_openvino_backend()
        rec_option.use_openvino_backend()

    elif args.backend.lower() == "pplite":
        assert args.device.lower(
        ) == "cpu", "Paddle Lite backend require inference on device CPU."
        det_option.use_lite_backend()
        cls_option.use_lite_backend()
        rec_option.use_lite_backend()

    return det_option, cls_option, rec_option

args = parse_arguments()

det_model_file = os.path.join(args.det_model, "inference.pdmodel")
det_params_file = os.path.join(args.det_model, "inference.pdiparams")

cls_model_file = os.path.join(args.cls_model, "inference.pdmodel")
cls_params_file = os.path.join(args.cls_model, "inference.pdiparams")

rec_model_file = os.path.join(args.rec_model, "inference.pdmodel")
rec_params_file = os.path.join(args.rec_model, "inference.pdiparams")
rec_label_file = args.rec_label_file

det_option, cls_option, rec_option = build_option(args)

det_model = fd.vision.ocr.DBDetector(
    det_model_file, det_params_file, runtime_option=det_option)

cls_model = fd.vision.ocr.Classifier(
    cls_model_file, cls_params_file, runtime_option=cls_option)

rec_model = fd.vision.ocr.Recognizer(
    rec_model_file, rec_params_file, rec_label_file, runtime_option=rec_option)

# Parameters settings for pre and post processing of Det/Cls/Rec Models.
# All parameters are set to default values.
det_model.preprocessor.max_side_len = 960
det_model.postprocessor.det_db_thresh = 0.3
det_model.postprocessor.det_db_box_thresh = 0.6
det_model.postprocessor.det_db_unclip_ratio = 1.5
det_model.postprocessor.det_db_score_mode = "slow"
det_model.postprocessor.use_dilation = False
cls_model.postprocessor.cls_thresh = 0.9

# Create PP-OCRv3, if cls_model is not needed, just set cls_model=None .
ppocr_v3 = fd.vision.ocr.PPOCRv3(det_model=det_model, cls_model=cls_model, rec_model=rec_model)

# Set inference batch size for cls model and rec model, the value could be -1 and 1 to positive infinity.
# When inference batch size is set to -1, it means that the inference batch size
# of the cls and rec models will be the same as the number of boxes detected by the det model.
ppocr_v3.cls_batch_size = args.cls_bs
ppocr_v3.rec_batch_size = args.rec_bs


def ocr(im: np.ndarray) -> list:
    # Predict and reutrn the results
    result = ppocr_v3.predict(im)

    # vis result
    vis_im = fd.vision.vis_ppocr(im, result)
    cv2.imwrite("visualized_result.jpg", vis_im)

    detects = []
    for i, (box, text, rec_score, cls_label, cls_score) in enumerate(
        zip(result.boxes, result.text, result.rec_scores, result.cls_labels,result.cls_scores)
    ):
        detect = {}
        detect["box"]  = [box[0:2], box[2:4], box[4:6], box[6:8]]
        detect["text"] = text
        detect["rec_score"] = rec_score
        detect["cls_label"] = cls_label
        detect["cls_score"] = cls_score
        detects.append(detect)
    return detects


app = FastAPI()

class ImageUrl(BaseModel):
    url: str = Field(default="https://gitee.com/paddlepaddle/PaddleOCR/raw/release/2.6/doc/imgs/12.jpg", description="图片url或本地路径")


check = re.compile(
    r'^(?:http|ftp)s?://'   # http:// or https:// or ftp:// or ftps://
    r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|' #domain...
    r'localhost|'           # localhost...
    r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})' # ...or ip
    r'(?::\d+)?'            # optional port
    r'(?:/?|[/?]\S+)$', re.IGNORECASE)


def check_is_url(url: str) -> bool:
    """检查字符串是否为url"""
    res = check.match(url)
    if res is None:
        return False
    else:
        return True


# 查询参数 q 的类型为 str，默认值为 None，因此它是可选的
# http://127.0.0.1:8001/ocr
@app.post("/ocr_url")
async def ocr_url(image: ImageUrl = Body()):
    try:
        # Read the input image
        if check_is_url(image.url):
            resp      = urlopen(image.url)
            array     = np.asarray(bytearray(resp.read()), dtype="uint8")
            # imdecode: In the case of color images, the decoded images will have the channels stored in **B G R** order.
            im = cv2.imdecode(array, cv2.IMREAD_COLOR)
        else:
            im = cv2.imread(image.url) # 本地图片

        detects = ocr(im)
    except:
        detects = []

    print(detects)
    return detects


ALLOW_SUFFIXES = [".jpg", ".jpeg", ".png", ".fig", ".tiff", ".webp"]
@app.post("/ocr_image")
async def ocr_image(file: UploadFile = File(description="A Pic"),):
    # suffix
    filename = Path(file.filename)
    suffix = filename.suffix
    if suffix not in ALLOW_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"suffix must be in {ALLOW_SUFFIXES}",
        )

    contents = await file.read()                        # async read

    # 转化为numpy数组再保存
    array    = np.asarray(bytearray(contents))          # 转化为1维数组
    im       = cv2.imdecode(array, cv2.IMREAD_COLOR)    # 转换为图片

    detects = ocr(im)

    print(detects)
    return detects


# run: uvicorn main:app --reload --port=8001
#   main: main.py 文件(一个 Python「模块」)。
#   app: 在 main.py 文件中通过 app = FastAPI() 创建的对象。
#   --reload: 让服务器在更新代码后重新启动。仅在开发时使用该选项。
if __name__ == "__main__":
    from pathlib import Path
    file = Path(__file__).stem  # get file name without suffix
    uvicorn.run(app=f"{file}:app", host="127.0.0.1", port=8001, reload=True)
