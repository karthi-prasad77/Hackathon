from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, status, Request
import cv2
import io
import matplotlib.pyplot as plt
from matplotlib import ft2font
from starlette.responses import StreamingResponse
from model import Data, User
from database import database
from fastapi.security import OAuth2PasswordRequestForm
from hashing import Hash
from jwttoken import create_access_token
from fastapi.security import HTTPBasic, HTTPBearer, HTTPBasicCredentials, OAuth2PasswordBearer
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse


token_auth_scheme = HTTPBearer() 
oauth2 = OAuth2PasswordBearer(tokenUrl = "token")


# craete an object for the FastAPI
app = FastAPI()


app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory="templates")


# create an get method
@app.get("/")
async def root(token: str = Depends(token_auth_scheme)):
    result = token.credentials
    return "AI based image editing tool"

# create an user registration route
@app.post("/api/register")
async def create_user(request: User):
    hashed_pass = Hash.bcrypt(request.password)
    user_object = dict(request)
    user_object['password'] = hashed_pass
    user_id = database['users'].insert_one(user_object)
    return {"res": "created"}

# create an user login route
@app.post("/api/login")
async def login(request:OAuth2PasswordRequestForm = Depends()):
    user = database['users'].find_one({"username": request.username})
    if not user:
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = f'No user found with this {request.username} username')
    if not Hash.verify(user['password'], request.password):
        raise HTTPException(status_code = status.HTTP_404_NOT_FOUND, detail = f'Wromg username and password')
    access_token = create_access_token(data = {"sub": user['username']})
    return {"Access token": access_token, "Token type": "bearer"}



# create an method to receive the image from the user
@app.post("/api/files")
async def UploadImage(token: str = Depends(token_auth_scheme), file: bytes = File(...)):
    result = token.credentials
    with open('./Images_upload/image.jpg', 'wb') as image:
        image.write(file)
        image.close()
    image = cv2.imread('Images_upload\\image.jpg')
    image_resize = cv2.resize(image, None, fx = 0.5, fy = 0.5)
    img_clear = cv2.medianBlur(image_resize, 3)
    img_clear = cv2.medianBlur(img_clear, 3)
    img_clear = cv2.medianBlur(img_clear, 3)
    img_clear = cv2.edgePreservingFilter(img_clear, sigma_s = 5)
    img_filter = cv2.bilateralFilter(img_clear, 3, 10, 5)
    for bi in range(2):
        image_filter = cv2.bilateralFilter(img_filter, 3, 20, 10)

    for bi in range(3):
        image_filter = cv2.bilateralFilter(img_filter, 5, 30, 10)

    guassian_masks = cv2.GaussianBlur(img_filter, (7, 7), 2)

    img_sharp = cv2.addWeighted(img_filter, 1.5, guassian_masks, -0.5, 0)
    img_sharp = cv2.addWeighted(img_sharp, 1.4, guassian_masks, -0.6, 10)

    # plt.imshow(img_sharp[:,:,::-1])
    res, im_png = cv2.imencode(".png", img_sharp[:,:,::-1])
    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type = "image/png")

# create an method to get the image, brightness and contrast value
@app.post("/api/alter", response_model = Data)
async def alterImage(Contrast:float ,Brightness:float,file: bytes = File(...), token: str = Depends(token_auth_scheme)):
    result = token.credentials
    with open('./Alter_Img/image.jpg', 'wb') as photo:
        photo.write(file)
        photo.close()
    img = cv2.imread('Alter_Img\\image.jpg')
    reShape = cv2.resize(img, (250, 250))

    # contrast and brightness
    Contrast = Contrast
    Brightness = Brightness

    alter = cv2.convertScaleAbs(img, alpha = Contrast, beta = Brightness)

    rest, img_png = cv2.imencode(".png", alter)
    return StreamingResponse(io.BytesIO(img_png.tobytes()), media_type = "image/png")

# create an method to get the image and detect the objects in the image
@app.post("/api/object")
async def objectDetection(token: str = Depends(token_auth_scheme), file: bytes = File(...)):
    config_file = './yolo_v3/mobile_net_v3.pbtxt'
    frozen_model = './yolo_v3/frozen_v3.pb'

    # tensorflow object detection model
    model = cv2.dnn_DetectionModel(frozen_model, config_file)

    classLabels = []

    filename = './yolo_v3/yolo.txt'
    with open(filename, 'rt') as img:
        classLabels = img.read().rstrip('\n').split('\n')

    # model training
    model.setInputSize(320, 320)
    model.setInputScale(1.0 / 127.5)
    model.setInputMean((127.5, 127.5, 127.5))
    model.setInputSwapRB(True)

    # reading image from the user
    with open('./ObjectDetection/image.jpg', 'wb') as ph:
        ph.write(file)
        ph.close()
    im = cv2.imread('ObjectDetection\\image.jpg')
    reShape = cv2.resize(im, (320, 320))

    # Object Detection
    ClassIndex, Confidence, bbox = model.detect(im, confThreshold = 0.5)

    # plotting boxes
    font = 3
    fonts = cv2.FONT_HERSHEY_PLAIN

    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), Confidence.flatten(), bbox):
        cv2.rectangle(im, boxes, (0, 255, 0), 3)   # for RGB channels
        cv2.putText(im, classLabels[ClassInd - 1], (boxes[0] + 10, boxes[1] + 40), fonts, fontScale = font, color = (0, 0, 255), thickness = 4)
    
    sol = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    r, i_png = cv2.imencode(".png", sol)
    return StreamingResponse(io.BytesIO(i_png.tobytes()), media_type = "image/png")


@app.post("/upload")
def upload(request:Request, file: UploadFile = File(...)):
    try:
        contents = file.file.read()
        with open("uploaded_" + file.filename, "wb") as f:
            f.write(contents)
    except Exception:
        return {"message": "There was an error uploading the file"}
    finally:
        file.file.close()
    
        
    return {"message": f"Successfuly uploaded {file.filename}"}

@app.get("/index")
def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

