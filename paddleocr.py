import importlib
import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np

__dir__ = os.path.dirname(__file__)

sys.path.append(os.path.join(__dir__, ''))

# def _import_file(module_name, file_path, make_importable=False):
#     spec = importlib.util.spec_from_file_location(module_name, file_path)
#     module = importlib.util.module_from_spec(spec)
#     spec.loader.exec_module(module)
#     if make_importable:
#         sys.modules[module_name] = module
#     return module


# tools = _import_file('tools', os.path.join(__dir__, 'tools/__init__.py'), make_importable=True)
ppocr = importlib.import_module('ppocr', 'paddleocr')
ppstructure = importlib.import_module('paddleocr')

from tools.infer import predict_system
from ppocr.utils.logging import get_logger

logger = get_logger()
from ppocr.utils.utility import check_and_read, get_image_file_list
from ppocr.utils.network import maybe_download, download_with_progressbar, is_link, confirm_model_dir_url
from tools.infer.utility import draw_ocr, str2bool, check_gpu, init_args

# from ppstructure.utility import init_args, draw_structure_result
# from ppstructure.predict_system import StructureSystem, save_structure_res, to_excel

__all__ = [
    'PaddleOCR', 'draw_ocr', 'download_with_progressbar'
]

SUPPORT_DET_MODEL = ['DB']
VERSION = '2.7.0.3'
SUPPORT_REC_MODEL = ['CRNN', 'SVTR_LCNet']
BASE_DIR = os.path.expanduser("~/.paddleocr/")

DEFAULT_OCR_MODEL_VERSION = 'PP-OCRv4'
SUPPORT_OCR_MODEL_VERSION = ['PP-OCR', 'PP-OCRv2', 'PP-OCRv3', 'PP-OCRv4']
# DEFAULT_STRUCTURE_MODEL_VERSION = 'PP-StructureV2'
# SUPPORT_STRUCTURE_MODEL_VERSION = ['PP-Structure', 'PP-StructureV2']
MODEL_URLS = {
    'OCR': {
        'PP-OCRv4': {
            'det': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv3/english/en_PP-OCRv3_det_infer.tar',
                },
            },
            'rec': {
                'en': {
                    'url':
                        'https://paddleocr.bj.bcebos.com/PP-OCRv4/english/en_PP-OCRv4_rec_infer.tar',
                    'dict_path': './ppocr/utils/en_dict.txt'
                },
            },
        },
    }
}


def parse_args(mMain=True):
    import argparse
    parser = init_args()
    parser.add_help = mMain
    parser.add_argument("--lang", type=str, default='en')
    parser.add_argument("--det", type=str2bool, default=True)
    parser.add_argument("--rec", type=str2bool, default=True)
    parser.add_argument("--type", type=str, default='ocr')
    parser.add_argument(
        "--ocr_version",
        type=str,
        choices=SUPPORT_OCR_MODEL_VERSION,
        default='PP-OCRv4',
        help='OCR Model version, the current model support list is as follows: '
             '1. PP-OCRv4/v3 Support Chinese and English detection and recognition model, and direction classifier model'
             '2. PP-OCRv2 Support Chinese detection and recognition model. '
             '3. PP-OCR support Chinese detection, recognition and direction classifier and multilingual recognition model.'
    )
    parser.add_argument(
        "--structure_version",
        type=str,
        # choices=SUPPORT_STRUCTURE_MODEL_VERSION,
        default='PP-StructureV2',
        help='Model version, the current model support list is as follows:'
             ' 1. PP-Structure Support en table structure model.'
             ' 2. PP-StructureV2 Support ch and en table structure model.')

    for action in parser._actions:
        if action.dest in [
            'rec_char_dict_path', 'table_char_dict_path', 'layout_dict_path'
        ]:
            action.default = None
    if mMain:
        return parser.parse_args()
    else:
        inference_args_dict = {}
        for action in parser._actions:
            inference_args_dict[action.dest] = action.default
        return argparse.Namespace(**inference_args_dict)


def parse_lang(lang):
    latin_lang = [
        'af', 'az', 'bs', 'cs', 'cy', 'da', 'de', 'es', 'et', 'fr', 'ga', 'hr',
        'hu', 'id', 'is', 'it', 'ku', 'la', 'lt', 'lv', 'mi', 'ms', 'mt', 'nl',
        'no', 'oc', 'pi', 'pl', 'pt', 'ro', 'rs_latin', 'sk', 'sl', 'sq', 'sv',
        'sw', 'tl', 'tr', 'uz', 'vi', 'french', 'german'
    ]
    arabic_lang = ['ar', 'fa', 'ug', 'ur']
    cyrillic_lang = [
        'ru', 'rs_cyrillic', 'be', 'bg', 'uk', 'mn', 'abq', 'ady', 'kbd', 'ava',
        'dar', 'inh', 'che', 'lbe', 'lez', 'tab'
    ]
    devanagari_lang = [
        'hi', 'mr', 'ne', 'bh', 'mai', 'ang', 'bho', 'mah', 'sck', 'new', 'gom',
        'sa', 'bgc'
    ]
    if lang in latin_lang:
        lang = 'latin'
    elif lang in arabic_lang:
        lang = 'arabic'
    elif lang in cyrillic_lang:
        lang = 'cyrillic'
    elif lang in devanagari_lang:
        lang = 'devanagari'
    assert lang in MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION][
        'rec'], 'param lang must in {}, but got {}'.format(
        MODEL_URLS['OCR'][DEFAULT_OCR_MODEL_VERSION]['rec'].keys(), lang)
    if lang == 'ch':
        det_lang = 'ch'
    elif lang == 'structure':
        det_lang = 'structure'
    elif lang in ['en', 'latin']:
        det_lang = 'en'
    else:
        det_lang = 'ml'
    return lang, det_lang


def get_model_config(type, version, model_type, lang):
    if type == 'OCR':
        DEFAULT_MODEL_VERSION = DEFAULT_OCR_MODEL_VERSION
    # elif type == 'STRUCTURE':
    #     DEFAULT_MODEL_VERSION = DEFAULT_STRUCTURE_MODEL_VERSION
    else:
        raise NotImplementedError

    model_urls = MODEL_URLS[type]
    if version not in model_urls:
        version = DEFAULT_MODEL_VERSION
    if model_type not in model_urls[version]:
        if model_type in model_urls[DEFAULT_MODEL_VERSION]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error('{} models is not support, we only support {}'.format(
                model_type, model_urls[DEFAULT_MODEL_VERSION].keys()))
            sys.exit(-1)

    if lang not in model_urls[version][model_type]:
        if lang in model_urls[DEFAULT_MODEL_VERSION][model_type]:
            version = DEFAULT_MODEL_VERSION
        else:
            logger.error(
                'lang {} is not support, we only support {} for {} models'.
                format(lang, model_urls[DEFAULT_MODEL_VERSION][model_type].keys(
                ), model_type))
            sys.exit(-1)
    return model_urls[version][model_type][lang]


def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def check_img(img):
    if isinstance(img, bytes):
        img = img_decode(img)
    if isinstance(img, str):
        # download net image
        if is_link(img):
            download_with_progressbar(img, 'tmp.jpg')
            img = 'tmp.jpg'
        image_file = img
        img, flag_gif, flag_pdf = check_and_read(image_file)
        if not flag_gif and not flag_pdf:
            with open(image_file, 'rb') as f:
                img = img_decode(f.read())
        if img is None:
            logger.error("error in loading image:{}".format(image_file))
            return None
    if isinstance(img, np.ndarray) and len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    return img


class PaddleOCR(predict_system.TextSystem):
    def __init__(self, **kwargs):
        """
        paddleocr package
        args:
            **kwargs: other params show in paddleocr --help
        """
        params = parse_args(mMain=False)
        params.__dict__.update(**kwargs)
        assert params.ocr_version in SUPPORT_OCR_MODEL_VERSION, "ocr_version must in {}, but get {}".format(
            SUPPORT_OCR_MODEL_VERSION, params.ocr_version)
        params.use_gpu = check_gpu(params.use_gpu)

        if not params.show_log:
            logger.setLevel(logging.INFO)
        # self.use_angle_cls = params.use_angle_cls
        lang, det_lang = parse_lang(params.lang)

        # init model dir
        det_model_config = get_model_config('OCR', params.ocr_version, 'det',
                                            det_lang)
        params.det_model_dir, det_url = confirm_model_dir_url(
            params.det_model_dir,
            os.path.join(BASE_DIR, 'whl', 'det', det_lang),
            det_model_config['url'])
        rec_model_config = get_model_config('OCR', params.ocr_version, 'rec',
                                            lang)
        params.rec_model_dir, rec_url = confirm_model_dir_url(
            params.rec_model_dir,
            os.path.join(BASE_DIR, 'whl', 'rec', lang), rec_model_config['url'])
        # cls_model_config = get_model_config('OCR', params.ocr_version, 'cls',
        #                                     'ch')
        # params.cls_model_dir, cls_url = confirm_model_dir_url(
        #     params.cls_model_dir,
        #     os.path.join(BASE_DIR, 'whl', 'cls'), cls_model_config['url'])
        if params.ocr_version == 'PP-OCRv3':
            params.rec_image_shape = "3, 48, 320"
        else:
            params.rec_image_shape = "3, 32, 320"
        # download model if using paddle infer
        if not params.use_onnx:
            maybe_download(params.det_model_dir, det_url)
            maybe_download(params.rec_model_dir, rec_url)
            # maybe_download(params.cls_model_dir, cls_url)

        if params.det_algorithm not in SUPPORT_DET_MODEL:
            logger.error('det_algorithm must in {}'.format(SUPPORT_DET_MODEL))
            sys.exit(0)
        if params.rec_algorithm not in SUPPORT_REC_MODEL:
            logger.error('rec_algorithm must in {}'.format(SUPPORT_REC_MODEL))
            sys.exit(0)

        if params.rec_char_dict_path is None:
            params.rec_char_dict_path = str(
                Path(__file__).parent / rec_model_config['dict_path'])

        logger.debug(params)
        # init det_model and rec_model
        super().__init__(params)
        self.page_num = params.page_num

    def ocr(self, img, det=True, rec=True, cls=True):
        """
        ocr with paddleocr
        args：
            img: img for ocr, support ndarray, img_path and list or ndarray
            det: use text detection or not. If false, only rec will be exec. Default is True
            rec: use text recognition or not. If false, only det will be exec. Default is True
            cls: use angle classifier or not. Default is True. If true, the text with rotation of 180 degrees can be recognized. If no text is rotated by 180 degrees, use cls=False to get better performance. Text with rotation of 90 or 270 degrees can be recognized even if cls=False.
        """
        assert isinstance(img, (np.ndarray, list, str, bytes))
        if isinstance(img, list) and det == True:
            logger.error('When input a list of images, det must be false')
            exit(0)
        # if cls == True and self.use_angle_cls == False:
        #     logger.warning(
        #         'Since the angle classifier is not initialized, the angle classifier will not be uesd during the forward process'
        #     )

        img = check_img(img)
        # for infer pdf file
        if isinstance(img, list):
            if self.page_num > len(img) or self.page_num == 0:
                self.page_num = len(img)
            imgs = img[:self.page_num]
        else:
            imgs = [img]
        if det and rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, rec_res, _ = self.__call__(img, cls)
                tmp_res = [[box.tolist(), res]
                           for box, res in zip(dt_boxes, rec_res)]
                ocr_res.append(tmp_res)
            return ocr_res
        elif det and not rec:
            ocr_res = []
            for idx, img in enumerate(imgs):
                dt_boxes, elapse = self.text_detector(img)
                tmp_res = [box.tolist() for box in dt_boxes]
                ocr_res.append(tmp_res)
            return ocr_res
        else:
            ocr_res = []
            cls_res = []
            for idx, img in enumerate(imgs):
                if not isinstance(img, list):
                    img = [img]
                if self.use_angle_cls and cls:
                    img, cls_res_tmp, elapse = self.text_classifier(img)
                    if not rec:
                        cls_res.append(cls_res_tmp)
                rec_res, elapse = self.text_recognizer(img)
                ocr_res.append(rec_res)
            if not rec:
                return cls_res
            return ocr_res


def main():
    # for cmd
    args = parse_args(mMain=True)
    image_dir = args.image_dir
    if is_link(image_dir):
        download_with_progressbar(image_dir, 'tmp.jpg')
        image_file_list = ['tmp.jpg']
    else:
        image_file_list = get_image_file_list(args.image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(args.image_dir))
        return
    if args.type == 'ocr':
        engine = PaddleOCR(**args.__dict__)
    # elif args.type == 'structure':
    #     engine = PPStructure(**args.__dict__)
    else:
        raise NotImplementedError

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))
        if args.type == 'ocr':
            result = engine.ocr(img_path,
                                det=args.det,
                                rec=args.rec,
                                cls=args.use_angle_cls)
            if result is not None:
                for idx in range(len(result)):
                    res = result[idx]
                    for line in res:
                        logger.info(line)
        elif args.type == 'structure':
            img, flag_gif, flag_pdf = check_and_read(img_path)
            if not flag_gif and not flag_pdf:
                img = cv2.imread(img_path)

            # if args.recovery and args.use_pdf2docx_api and flag_pdf:
            #     from pdf2docx.converter import Converter
            #     docx_file = os.path.join(args.output, '{}.docx'.format(img_name))
            #     cv = Converter(img_path)
            #     cv.convert(docx_file)
            #     cv.close()
            #     logger.info('docx save to {}'.format(docx_file))
            #     continue

            if not flag_pdf:
                if img is None:
                    logger.error("error in loading image:{}".format(img_path))
                    continue
                img_paths = [[img_path, img]]
            else:
                img_paths = []
                for index, pdf_img in enumerate(img):
                    os.makedirs(
                        os.path.join(args.output, img_name), exist_ok=True)
                    pdf_img_path = os.path.join(
                        args.output, img_name,
                        img_name + '_' + str(index) + '.jpg')
                    cv2.imwrite(pdf_img_path, pdf_img)
                    img_paths.append([pdf_img_path, pdf_img])

            all_res = []
            for index, (new_img_path, img) in enumerate(img_paths):
                logger.info('processing {}/{} page:'.format(index + 1, len(img_paths)))
                new_img_name = os.path.basename(new_img_path).split('.')[0]
                result = engine(new_img_path, img_idx=index)
                # save_structure_res(result, args.output, img_name, index)

                # if args.recovery and result != []:
                #     from copy import deepcopy
                #     from ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
                #     h, w, _ = img.shape
                #     result_cp = deepcopy(result)
                #     result_sorted = sorted_layout_boxes(result_cp, w)
                #     all_res += result_sorted

            # if args.recovery and all_res != []:
            #     try:
            #         from ppstructure.recovery.recovery_to_doc import convert_info_docx
            #         convert_info_docx(img, all_res, args.output, img_name)
            #     except Exception as ex:
            #         logger.error("error in layout recovery image:{}, err msg: {}".format(img_name, ex))
            #         continue

            for item in all_res:
                item.pop('img')
                item.pop('res')
                logger.info(item)
            logger.info('result save to {}'.format(args.output))


if __name__ == '__main__':
    args = parse_args(mMain=True)

    image_dir = 'test'
    # print(image_dir)
    image_file_list = get_image_file_list(image_dir)
    if len(image_file_list) == 0:
        logger.error('no images find in {}'.format(image_dir))
        # return

    engine = PaddleOCR()

    for img_path in image_file_list:
        img_name = os.path.basename(img_path).split('.')[0]
        logger.info('{}{}{}'.format('*' * 10, img_path, '*' * 10))

        result = engine.ocr(img_path, det=True, rec=True, cls=True)
        if result is not None:
            for idx in range(len(result)):
                res = result[idx]
                for line in res:
                    logger.info(line)
        else:
            print("result is none")

        # 显示结果
        from PIL import Image

        result = result[0]
        image = Image.open(img_path).convert('RGB')

        boxes = [line[0] for line in result]
        txts = [line[1][0] for line in result]
        scores = [line[1][1] for line in result]

        im_show = draw_ocr(image, boxes, txts, scores)

        im_show = Image.fromarray(im_show)
        im_show.save('img/{}.jpg'.format(img_name))
