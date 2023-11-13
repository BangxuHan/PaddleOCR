import base64
import json
import cv2
import numpy as np
import copy
import time

from flask import Flask, request
from flask_restful import Resource, Api
from tools.infer import predict_system
import tools.infer.utility as utility
from tools.infer.utility import get_rotate_crop_image
from ppocr.data import create_operators, transform
app = Flask("paddleOCR")
api = Api(app)

SEQ_LEN = 16


def sorted_boxes(dt_boxes):
    """
    Sort text boxes in order from top to bottom, left to right
    args:
        dt_boxes(array):detected text boxes with shape [4, 2]
    return:
        sorted boxes(array) with shape [4, 2]
    """
    num_boxes = dt_boxes.shape[0]
    sorted_boxes = sorted(dt_boxes, key=lambda x: (x[0][1], x[0][0]))
    _boxes = list(sorted_boxes)

    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            if abs(_boxes[j + 1][0][1] - _boxes[j][0][1]) < 10 and \
                    (_boxes[j + 1][0][0] < _boxes[j][0][0]):
                tmp = _boxes[j]
                _boxes[j] = _boxes[j + 1]
                _boxes[j + 1] = tmp
            else:
                break
    return _boxes


class TextRecognition(object):
    def __init__(self, args, seq_len):
        self.number_merge = False
        self.args = args
        self.seq_len = SEQ_LEN

        args.det_model_dir = 'pretrain/en_PP-OCRv3_det_infer'
        args.rec_model_dir = 'output/v3_en_mobile_infer'

        self.pred = predict_system.TextSystem(args)

        self.drop_score = args.drop_score

    def __call__(self, img):
        time_dict = {'det': 0, 'rec': 0, 'cls': 0, 'all': 0}
        start = time.time()
        ori_im = img.copy()
        dt_boxes, elapse = self.pred.text_detector(img)
        time_dict['det'] = elapse

        if dt_boxes is None:
            return None, None
        dt_boxes = sorted_boxes(dt_boxes)

        img_crop_list = []
        for bno in range(len(dt_boxes)):
            tmp_box = copy.deepcopy(dt_boxes[bno])
            img_crop = get_rotate_crop_image(ori_im, tmp_box)

            img_crop_list.append(img_crop)

        rec_res, elapse = self.pred.text_recognizer(img_crop_list)
        time_dict['rec'] = elapse

        filter_boxes, filter_rec_res = [], []
        for box, rec_result in zip(dt_boxes, rec_res):
            text, score = rec_result
            if score >= self.drop_score:
                filter_boxes.append(box)
                filter_rec_res.append(rec_result)

        end = time.time()
        time_dict['all'] = end - start
        return filter_boxes, filter_rec_res, time_dict

    def predict(self, imgs):
        # imgs = [img]
        ocr_res = []
        for idx, img in enumerate(imgs):
            dt_boxes, rec_res, _ = self.__call__(img)
            tmp_res = [[box.tolist(), res] for box, res in zip(dt_boxes, rec_res)]

            if self.number_merge:
                res_part, res_com = '', ''
                score_com = 0.0
                for res in rec_res:
                    text, score = res
                    if not res_com:
                        if text.isalpha() and len(text) == 1:
                            res_part = text.upper()
                            score_com = score
                        if res_part and text.isdigit() and len(text) >= 4:
                            res_com = res_part + text
                            score_com = (score_com + score) / 2
                        if text.isdigit() and not res_part and len(text) >= 4:
                            res_com = text
                            score_com = score

                rec_res = res_com, score_com

            ocr_res.append(rec_res)
        return ocr_res


class CharRecognition(Resource):
    def post(self):
        temp = request.get_data(as_text=True)
        data = json.loads(temp)
        images = data['image']
        imagebuf = []
        for imagestr in images:
            imagedata_base64 = base64.b64decode(imagestr)
            np_arr = np.frombuffer(imagedata_base64, dtype=np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            # image = image_preprocessing(image, (cfg.image_size, cfg.image_size))
            imagebuf.append(image)
        imagebuf = np.array(imagebuf)
        ocr_res = model.predict(imagebuf)

        # for i in range(nlen):
        #     temp = {
        #         # "words": boxes[i],
        #         "probability": score[i]
        #     }
        #     words_res.append(temp)

        result = {"words_result_num": len(ocr_res),
                  "words_result": ocr_res
                  }
        return app.response_class(json.dumps(result), mimetype='application/json')


api.add_resource(CharRecognition, '/charrecog')
model = TextRecognition(utility.parse_args(), SEQ_LEN + 1)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
