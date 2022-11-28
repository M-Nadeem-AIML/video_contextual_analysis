import time
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image
import whisper
import argparse
import glob
import os
import numpy as np
import json
import cv2

f = open("audio.txt", "a")
f1= open("video.txt", "a")


try:
    os.remove('prediction.txt')
except:
    pass
removing_files = glob.glob('./*.jpg')
for i in removing_files:
    os.remove(i)
   
width = 512
height = 512
dim = (width, height)
A = np.zeros((height,width,3), np.uint8)
# vidcap = cv2.VideoCapture('video.mp4')
count=1
def getFrame(sec):
    global A
    global vidcap1
    global count
    # cv2.VideoCapture('video.mp4')
    vidcap1.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = vidcap1.read()
    if hasFrames:
        image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        # cv2.imwrite("image"+str(count)+".jpg", image) 
        B=image
        errorL2 = cv2.norm(A, B, cv2.NORM_L2)
        similarity = 1 - errorL2 / ( height * width )
        A=B
        if similarity<=0.75:
               cv2.imwrite("image" +str(count)+".jpg", image) 
               print('Similarity = ',similarity)
               
        
    return hasFrames




model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
feature_extractor = ViTFeatureExtractor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
max_length = 16
num_beams = 4
gen_kwargs = {"max_length": max_length, "num_beams": num_beams}

def predict_step(image_paths):
  images = []
  for image_path in image_paths:
    i_image = Image.open(image_path)
    if i_image.mode != "RGB":
      i_image = i_image.convert(mode="RGB")

    images.append(i_image)

  pixel_values = feature_extractor(images=images, return_tensors="pt").pixel_values
  pixel_values = pixel_values.to(device)

  output_ids = model.generate(pixel_values, **gen_kwargs)

  preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
  preds = [pred.strip() for pred in preds]
  return preds

# print(predict_step(['images.jpeg']))
# end=time.time()
# print(end-start)


def audio():
    model = whisper.load_model("base")
    result = model.transcribe("Test_2.mp4")
    print(result["text"])
    f.write(result["text"])
    f.close()






def main():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--img', type=str,
    #                     help='Image path')
    parser.add_argument('--video', type=str,
                       help='Image path')
    parser.add_argument('--img-dir', type=str, default='./',
                        help='Image directory path, instead of a single image')
    args = parser.parse_args()

    model = whisper.load_model("base")
    result = model.transcribe(args.video)
    print(result["text"])
    f.write(result["text"])
    f.close()

    global vidcap1
    global count

    vidcap=args.video
    vidcap1=cv2.VideoCapture(vidcap)

    print(vidcap1)
    sec = 0
    frameRate = 3
    success = getFrame(sec)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec)
    if args.img_dir:  # Read all images in directory
        img_paths = [
            i for i in glob.glob(os.path.join(args.img_dir, '*')) if
            i.endswith(('png', 'jpg'))]
        img_paths = sorted(img_paths)
        for image in img_paths:
            result=predict_step([image])
            f1.write(result[0])
    else:  # Load a single image
        img_paths = [args.img]
 

if __name__ == '__main__':
    main()