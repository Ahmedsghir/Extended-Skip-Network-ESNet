import torch
import os
import time
import cv2
from utils import load_checkpoint, preprocess, postprocess
from ESNet_model import ESNet

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_HEIGHT = 352
IMAGE_WIDTH = 640
INPUT_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
DATASET_PATH = "inference/testing_images/"
RESULTS_DIR = 'inference/results/'
PRETRAINED_MODEL_PATH = 'model.pth.tar'


def main():
    print("[INFO] loading model...")
    model = ESNet(in_ch=3, n_class=1).to(DEVICE)
    load_checkpoint(torch.load(PRETRAINED_MODEL_PATH), model)
    model.eval()

    for image_name in os.listdir(DATASET_PATH):
        if image_name.endswith(".png"):
            print("[INFO] processing: " + image_name)
            full_path = os.path.join(DATASET_PATH, image_name)
            testing_image = cv2.imread(full_path)
            t_img = preprocess(testing_image, INPUT_SIZE)
            start = time.time()
            with torch.no_grad():
                preds = torch.sigmoid(model(t_img))
                preds = (preds > 0.5)
                end = time.time()
                run_time = end - start
            print("[INFO] inference took {:.4f} seconds".format(run_time))

            output, final_pred_mask = postprocess(testing_image, preds, INPUT_SIZE)
            output_name = "output_" + image_name
            results_path = os.path.join(RESULTS_DIR, output_name)
            print("mask size: ", final_pred_mask.shape)

            cv2.imwrite(results_path, output)


if __name__ == "__main__":
    main()
