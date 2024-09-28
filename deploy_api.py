from fastapi import FastAPI, UploadFile, Form,Query
from hivision import IDCreator
from hivision.error import FaceError
from hivision.creator.layout_calculator import (
    generate_layout_photo,
    generate_layout_image,
)
from hivision.creator.choose_handler import choose_handler
from hivision.utils import add_background, resize_image_to_kb_base64, hex_to_rgb
import base64
import numpy as np
import cv2
from database.client import db
from database.response import CRUD
from bson.json_util import dumps
from datetime import datetime
from pymongo import ReturnDocument
from settings import WXURL, APPID, SECRET
import requests


app = FastAPI()
creator = IDCreator()

collection = db['photos']
collection_user = db['user']
# 将图像转换为Base64编码
def numpy_2_base64(img: np.ndarray):
    retval, buffer = cv2.imencode(".png", img)
    base64_image = base64.b64encode(buffer).decode("utf-8")

    return base64_image


# 证件照智能制作接口
@app.post("/idphoto")
async def idphoto_inference(
    input_image: UploadFile,
    height: str = Form(...),
    width: str = Form(...),
    human_matting_model: str = Form("hivision_modnet"),
    face_detect_model: str = Form("mtcnn"),
    head_measure_ratio=0.2,
    head_height_ratio=0.45,
    top_distance_max=0.12,
    top_distance_min=0.10,
):

    image_bytes = await input_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, human_matting_model, face_detect_model)

    # 将字符串转为元组
    size = (int(height), int(width))
    try:
        result = creator(
            img,
            size=size,
            head_measure_ratio=head_measure_ratio,
            head_height_ratio=head_height_ratio,
        )
    except FaceError:
        result_message = {"status": False}
    # 如果检测到人脸数量等于1, 则返回标准证和高清照结果（png 4通道图像）
    else:
        result_message = {
            "status": True,
            "image_base64_standard": numpy_2_base64(result.standard),
            "image_base64_hd": numpy_2_base64(result.hd),
        }

    return result_message


# 人像抠图接口
@app.post("/human_matting")
async def idphoto_inference(
    input_image: UploadFile,
    human_matting_model: str = Form("hivision_modnet"),
):
    image_bytes = await input_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # ------------------- 选择抠图与人脸检测模型 -------------------
    choose_handler(creator, human_matting_model, None)

    try:
        result = creator(
            img,
            change_bg_only=True,
        )
    except FaceError:
        result_message = {"status": False}

    else:
        result_message = {
            "status": True,
            "image_base64": numpy_2_base64(result.standard),
        }
    return result_message


# 透明图像添加纯色背景接口
@app.post("/add_background")
async def photo_add_background(
    input_image: UploadFile,
    color: str = Form(...),
    kb: str = Form(None),
    render: int = Form(0),
):
    render_choice = ["pure_color", "updown_gradient", "center_gradient"]

    image_bytes = await input_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    color = hex_to_rgb(color)
    color = (color[2], color[1], color[0])

    result_image = add_background(
        img,
        bgr=color,
        mode=render_choice[render],
    ).astype(np.uint8)

    if kb:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_image_base64 = resize_image_to_kb_base64(result_image, int(kb))
    else:
        result_image_base64 = numpy_2_base64(result_image)

    # try:
    result_messgae = {
        "status": True,
        "image_base64": result_image_base64,
    }

    # except Exception as e:
    #     print(e)
    #     result_messgae = {
    #         "status": False,
    #         "error": e
    #     }

    return result_messgae



# 六寸排版照生成接口
@app.post("/generate_layout_photos")
async def generate_layout_photos(
    input_image: UploadFile,
    height: str = Form(...),
    width: str = Form(...),
    kb: str = Form(None),
):
    # try:
    image_bytes = await input_image.read()
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    size = (int(height), int(width))

    typography_arr, typography_rotate = generate_layout_photo(
        input_height=size[0], input_width=size[1]
    )

    result_layout_image = generate_layout_image(
        img, typography_arr, typography_rotate, height=size[0], width=size[1]
    ).astype(np.uint8)

    if kb:
        result_layout_image = cv2.cvtColor(result_layout_image, cv2.COLOR_RGB2BGR)
        result_layout_image_base64 = resize_image_to_kb_base64(
            result_layout_image, int(kb)
        )
    else:
        result_layout_image_base64 = numpy_2_base64(result_layout_image)

    result_messgae = {
        "status": True,
        "image_base64": result_layout_image_base64,
    }

    # except Exception as e:
    #     result_messgae = {
    #         "status": False,
    #     }

    return result_messgae

# <!----------------------------------自己写的接口------自己写的接口-----------自己写的接口-------自己写的接口------------------------------->


# 透明图像添加纯色背景接口 传递图像是base64格式
@app.post("/v1/add_background")
async def v1_photo_add_background(
    input_image_base64: str = Form(...),
    color: str = Form(...),
    kb: str = Form(None),
    render: int = Form(0),
):
    render_choice = ["pure_color", "updown_gradient", "center_gradient"]
    image_bytes = base64.b64decode(input_image_base64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)

    color = hex_to_rgb(color)
    color = (color[2], color[1], color[0])

    result_image = add_background(
        img,
        bgr=color,
        mode=render_choice[render],
    ).astype(np.uint8)

    if kb:
        result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
        result_image_base64 = resize_image_to_kb_base64(result_image, int(kb))
    else:
        result_image_base64 = numpy_2_base64(result_image)

    # try:
    result_messgae = {
        "status": True,
        "image_base64": result_image_base64,
    }

    # except Exception as e:
    #     print(e)
    #     result_messgae = {
    #         "status": False,
    #         "error": e
    #     }

    return result_messgae



@app.get("/photo_size")
async def test(name: str = Query(default=None, title='证件照名称'),category: int = Query(default=None, title='证件照类型')
               ,recommend: int = Query(default=None, title='是否热门')):
    condition = {}
    if name is not None and name != '':
        condition['name'] = {'$regex': name, '$options': 'i'}
    if category is not None and category != '':
        condition['category'] = category
    if recommend is not None and recommend != '':
        condition['recommend'] = recommend

    preview_data_list = list(collection.find(condition).sort('id').limit(200))

    result_preview_data_list = []
    for preview_data in preview_data_list:
        result_preview_data = {
            'id': str(preview_data['_id']),
            'name': preview_data['name'],
            'width': preview_data['width_mm'],
            'height': preview_data['height_mm'],
            'pix_width': preview_data['width_px'],
            'pix_height': preview_data['height_px'],
            'category': preview_data['category']
        }
        result_preview_data_list.append(result_preview_data)


    return CRUD(status=True, msg='请求成功', data=result_preview_data_list)


@app.get("/login")
async def test(code: str = Query(default=None, title='微信code')):
    url=WXURL+'?appid='+APPID+'&secret='+SECRET+'&js_code='+code+'&grant_type=authorization_code'

    response = requests.get(url)
    response_json= response.json()
    openid= response_json['openid']
    # 尝试查找是否有相应的 code 并更新登录时间与登录次数
    result =  collection_user.find_one_and_update(
        {"openid": openid},  # 查询条件
        {
            "$set": {"last_login": datetime.utcnow()},  # 更新最后登录时间
            "$inc": {"login_count": 1}  # 登录次数加 1
        },
        upsert=False,  # 不自动插入
        return_document=ReturnDocument.AFTER
    )

    if result:
        # 如果找到并更新了对应的文档
        return CRUD(status=True, msg='登录成功', data=dumps(result))
    else:
        # 如果没有找到对应的文档，插入一条新的记录
        new_document = {
            "openid": openid,
            "created_at": datetime.utcnow(),
            "last_login": datetime.utcnow(),
            "login_count": 1
        }
        insert_result =  collection_user.insert_one(new_document)
        if insert_result.inserted_id:
            return CRUD(status=True, msg='首次登录成功', data=dumps(new_document))

# 测试接口
@app.get("/test")
async def test():

    result_messgae = {
        "status": "123",
        "image_base64": "123",
    }


    return result_messgae

# 测试接口
@app.post("/qtest")
async def test():

    result_messgae = {
        "status": "123",
        "image_base64": "123",
    }
    return result_messgae


if __name__ == "__main__":
    import uvicorn

    # 在8080端口运行推理服务
    uvicorn.run(app, host="0.0.0.0", port=8080)
