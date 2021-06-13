import cv2
import numpy as np
import os
import glob

if cv2.__version__ != "4.5.2":
    print("please install opencv-python==4.5")

'''
!pip install opencv-python==4.5
'''


def remove_watermark(img, kernel=None):
    """
    kernel must be odd number tuple
    """
    if kernel is None:
        kernel = (5, 5)

    if len(kernel) == 2:
        denoise = cv2.fastNlMeansDenoisingColored(img, None, kernel[0], kernel[1], 7, 21)
        return denoise
    else:
        print("kernel must have 2 values")


def remove_logo(img):
    # Load Yolo
    weights = "resourses/custom-yolov4-detector_best.weights"
    config = "resourses/custom-yolov4-detector.cfg"

    net = cv2.dnn.readNet(weights, config)
    # get layers of the network
    layer_names = net.getLayerNames()
    # Determine the output layer names from the YOLO model
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # read image
    # img = cv2.imread(img)

    # img = cv2.resize(img, None, fx=0.4, fy=0.4)
    height, width, channels = img.shape

    # USing blob function of opencv to preprocess image
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)
    # Detecting objects
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    # We use NMS function in opencv to perform Non-maximum Suppression
    # we give it score threshold and nms threshold as arguments.
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            # print(x, y, w, h)

            # crop detected part
            crop = img[y:y + h, x:x + w]

            # convert crop to gray
            grayImage = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

            # draw edges on the logo to highlight the roi
            grayImage = cv2.Canny(grayImage, 100, 200)

            # Blur make roi bigger for mask
            grayImage = cv2.GaussianBlur(grayImage, (5, 5), 0)

            # convert roi into binary image
            thresh, logoframe = cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY)

            # make highlighted part bigger
            logoframe = cv2.dilate(logoframe,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

            # create blank black mask with same height and width as img
            maskframe = np.zeros((height, width), np.uint8)

            # paste logo on the mask
            maskframe[y:y + h, x:x + w] = logoframe

            # remove logo from original image using mask
            cleaned_logo = cv2.inpaint(img, maskframe, 3, cv2.INPAINT_NS)

    return cleaned_logo
    # cv2.imwrite("detected.jpg", img)
    # cv2.imwrite("crop.jpg", final)
    # cv2.waitKey(0)


def post_processing(img):
    alpha = 2.0
    beta = -160

    try:
        new = alpha * img + beta
        new = np.clip(new, 0, 255).astype(np.uint8)
        dst = cv2.detailEnhance(new, sigma_s=10, sigma_r=0.15)
        return dst
    except:
        return img


def process_image(input_dir = None, output_dir = None, single_image = None, kernel_size = None):
    # input folder
    # output folder
    # single image
    # kernel size
    print(output_dir,output_dir,single_image,kernel_size)
    # file extensions
    extensions = ("*.jpg", "*.png")
    image_list = []

    # check if input dir is valid or not
    if input_dir is not None:
        if os.path.isdir(input_dir):
            for extension in extensions:
                image_list.extend(glob.glob(os.path.join(input_dir,extension)))
        else:
            print("Please provide valid input directory". input_dir)

    if output_dir is not None:
        if not os.path.isdir(output_dir):
            print("Please provide valid output directory". output_dir)

    if single_image is not None:
        if os.path.isfile(single_image):
            image_list.extend([single_image])
        else:
            print("Please provide valid image". single_image)

    if kernel_size is not None:
        if len(kernel_size) == 2:
            if kernel_size[0] % 2 == 0 and kernel_size[1] % 2 == 0:
                print("Please provide valid kernel values". kernel_size)

    print(image_list)
    # Process Image
    for image in image_list:
        img = cv2.imread(image)

        # remove watermark
        cleaned_watermark = remove_watermark(img, kernel_size)

        # remove logo
        cleaned_logo = remove_logo(cleaned_watermark)

        # post processing
        final = post_processing(cleaned_logo)

        # image name
        image_name = os.path.split(image)[1]

        # write image to output dir
        cv2.imwrite(os.path.join(output_dir, image_name), final)
        print(output_dir)




"""
def process_image(input_dir: Any = None,
               output_dir: Any = None,
               single_image: Any = None,
               kernel_size: Any = None) -> None
"""

# process_image("INPUT_DIR", "OUTPUT_DIR", None, (5,5))