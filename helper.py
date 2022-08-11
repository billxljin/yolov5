from utils.plots import plot_results
import os,sys
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
# plot_results('C:/Users/Administrator/Desktop/results.csv')  # plot 'results.csv' as 'results.png'


import cv2
import torch
from PIL import Image





def main():
    imgfolder = sys.argv[1]
    modeldir = sys.argv[2]
    savedir = sys.argv[3]

    model = torch.hub.load('ultralytics/yolov5', 'custom', path='D:/Deeplearning/YOLO5/yolov5/runs/train/exp16 -single class relabel/weights/best.pt')
    # model.cuda()

    for filename in os.listdir(imgfolder):
      imghdr = os.path.join(imgfolder,filename)
      img = cv2.imread(imghdr)[..., ::-1]  # OpenCV image (BGR to RGB)
      results = model(img, size=1408)
      results.print()
      # results.show()


if __name__ == "__main__":
    # draw_yolo_label()
    main()


# python D:\Deeplearning\YOLO5\yolov5\helper.py  "X:\Lvran\05-30 500W VS 1200W rgb\1200w\Yolov5\train\S1\test\images" "D:/Deeplearning/YOLO5/yolov5/runs/train/exp16 -single class relabel/weights/best.pt"   " s"