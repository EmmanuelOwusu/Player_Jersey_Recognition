import torch ,  easyocr
import os,numpy as np
from YouGet import download as file_downloader
import cv2
import Config
config= Config.Configurations()
cd = os.curdir
ocr = easyocr.Reader(['en'],gpu=False)
urls = ["https://drive.google.com/file/d/1mQ49zQdL3NSxxXOoMy-mT-RTaHIml2XF/view?usp=sharing" ,
        "https://drive.google.com/file/d/1zKOPjD--hbC4sbIJhjPh3wHxxgkkgjIs/view?usp=sharing" ,
        "https://drive.google.com/file/d/1rLgOmqTGKRdpfCDY2i949GOXwqsGk-fA/view?usp=sharing",
        ]
        # [ball, person,jersey_number]


path_location = config.workin_dir+'/ObjectDetection/models'
filename = ["ball", "person", "jersey_number"]
model= []
for i, url in enumerate(urls):

    if os.path.exists(path_location+'/'+filename[i]+".pt"):
        print("model file for {} detection exists".format(filename[i]))
    else:
        file_downloader(url, filename=filename[i], path_location=path_location, fileformat='.pt')
        print(config.workin_dir +'/ObjectDetection/yolov5')
        print('downloaded model file for {} detection...'.format(filename[i]))
    print(config.workin_dir + '/ObjectDetection/yolov5')
    model.append(torch.hub._load_local(config.workin_dir +'/ObjectDetection/yolov5', 'custom', path='{}/{}.pt'.format(path_location,filename[i])))  # local model # local model

if "weights" in os.listdir(config.workin_dir):
    print("weight for perspective transform present")
else:
    file_downloader("https://drive.google.com/file/d/11xRV2yjQ1FtasDHEhUAV5S9Aiou-K58I/view?usp=sharing","weights",config.workin_dir,".zip")
    os.system("unzip weights.zip")


def ocr_it(npimg):
    img = cv2.resize(npimg, (224,224))
    try:

        text = ocr.readtext(img)    
        #text = ocr.readtext(img,  allowlist ='0123456789')
        return str(text[0][1])
    except:
        return "?"


def ball(npimg):
    original = npimg
    imgs = [npimg[..., ::-1]]  # batch of images
    # Inference
    results = model[0](imgs).xyxy[0].cpu().numpy()
    ball_con = {'bb': [], 'con': [], 'class': [], 'cropped': []}
    if len(results) > 0:
        confidences = [c[4] for c in results]
        index = np.argmax(confidences)
        x, y, x1, y1 = [round(i) for i in results[index][:4]]
        ball_con['bb'] = [x, y, x1-x, y1-y] #x, y, w, h
        ball_con['class'] = 1
        ball_con['con'] = confidences[np.argmax(confidences)]
        ball_con['cropped'].append(original[y:y1, x:x1])
        return ball_con
    else:
        return None


def person(npimg):
    original = npimg
    imgs = [npimg[..., ::-1]]
    # Inference
    results = model[1](imgs).xyxy[0].cpu().numpy()
    allperson = {'bb': [], 'confidence': [], 'class': [], 'cropped': []}
    if len(results)> 0:
        for p in results:
            if p[-1] == 0:
                x, y, x1, y1 = [round(i) for i in p[:4]]
                allperson["bb"].append([x, y, x1-x, y1-y])
                allperson["confidence"].append(p[4])
                allperson["class"].append(p[-1])
                allperson["cropped"].append(original[y:y1, x:x1])
        return allperson
    else:
        return None


def jersey_number(npimg):
    original = npimg
    imgs = [npimg[..., ::-1]]  # batch of images
    # Inference
    results = model[2](imgs).xyxy[0].cpu().numpy()
    jersey_info = {'bb': None, 'con': None, 'class': None, 'cropped': None, "text": None}
    if len(results) > 0:
        confidences = [c[4] for c in results]
        index = np.argmax(confidences)
        x, y, x1, y1 = [round(i) for i in results[index][:4]]
        jersey_info['bb'] = [x, y, x1-x, y1-y]
        x,y,w,h = jersey_info['bb']
        jersey_info['class'] = 3
        jersey_info['con'] = confidences[np.argmax(confidences)]
        jersey_info['cropped']=(original[y:y+h, x:x+w])
        jersey_info['text'] = ocr_it(jersey_info['cropped'])
        return jersey_info
    else:
        return None



def roi(npimg):
    return person(npimg), ball(npimg)


def plot_one_box(frame, bb, color=(128, 128, 128), label=None, line_thickness=3):
    x,y,w,h = bb
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    frame = cv2.rectangle(frame, pt1, pt2, color, 2)
    frame = cv2.putText(frame, label, pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, line_thickness)
    return frame


def plot_many_box(frame, bbs, color=(128, 128, 128), label=None, line_thickness=3):
    if label==None:
        label = label*len(bbs)
    for i, bb in enumerate(bbs):
        x, y, w, h = bb
        pt1 = (x, y)
        pt2 = (x + w, y + h)
        frame = cv2.rectangle(frame, pt1, pt2, color, 2)
        frame = cv2.putText(frame, label[i], pt1, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, line_thickness)
    return frame


def local_2_global(local_point,inner_bb):
    x,y,w,h=local_point
    i_x,i_y, i_w, i_h =inner_bb
    X = x+i_x
    Y = y+i_y

    X2 = (x+w)+(i_x)
    Y2 = (y+h)+(i_y)


    return X,Y,X2,Y2
