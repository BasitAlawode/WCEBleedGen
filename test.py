import os
import torch
import cv2
import numpy as np
import argparse
import yaml
import glob
import os
import time
import torchinfo
import pandas as pd
from natsort import natsorted

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

test_images_path = 'train/Val'
#test_images_path = 'test/Test_Dataset_1'
#test_images_path = 'test/Test_Dataset_2'

d_set_name = test_images_path.split('/')[-1]
config_path = 'bleedgen.yaml'

from code_utils.vision_transformers.detection.detr.model import DETRModel
from code_utils.tools.utils.detection.detr.general import (
    set_infer_dir,
    load_weights
)
from code_utils.tools.utils.detection.detr.transforms import infer_transforms, resize
from code_utils.tools.utils.detection.detr.annotations import (
    convert_detections,
    inference_annotations, 
)

np.random.seed(2023)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-w', 
        '--weights',
    )
    parser.add_argument(
        '-i', '--input', 
        help='folder path to input input image (one image or a folder path)',
    )
    parser.add_argument(
        '--data', 
        default=None,
        help='(optional) path to the data config file'
    )
    parser.add_argument(
        '--model', 
        default='detr_resnet50',
        help='name of the model'
    )
    parser.add_argument(
        '--device', 
        default=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
        help='computation/training device, default is GPU if GPU present'
    )
    parser.add_argument(
        '--imgsz', 
        '--img-size',
        default=640,
        dest='imgsz',
        type=int,
        help='resize image to, by default use the original frame/image size'
    )
    parser.add_argument(
        '-t', 
        '--threshold',
        type=float,
        default=0.5,
        help='confidence threshold for visualization'
    )
    parser.add_argument(
        '--name', 
        default=None, 
        type=str, 
        help='training result dir name in outputs/training/, (default res_#)'
    )
    parser.add_argument(
        '--hide-labels',
        dest='hide_labels',
        action='store_true',
        help='do not show labels during on top of bounding boxes'
    )
    parser.add_argument(
        '--show', 
        dest='show', 
        action='store_true',
        help='visualize output only if this argument is passed'
    )
    parser.add_argument(
        '--track',
        action='store_true'
    )
    parser.add_argument(
        '--classes',
        nargs='+',
        type=int,
        default=None,
        help='filter classes by visualization, --classes 1 2 3'
    )
    args = parser.parse_args()
    return args

def collect_all_images(dir_test):
    """
    Function to return a list of image paths.
    :param dir_test: Directory containing images or single image path.
    Returns:
        test_images: List containing all image paths.
    """
    test_images = []
    if os.path.isdir(dir_test):
        image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm']
        for file_type in image_file_types:
            test_images.extend(glob.glob(f"{dir_test}/{file_type}"))
    else:
        test_images.append(dir_test)
    return test_images   

def main(args):
    
    from train import model_backbone, img_size
    args.name = model_backbone
    args.model = model_backbone
    args.data = config_path
    args.imgsz = img_size
    args.weights = f"trained_weights/best_model.pth"
    args.input = test_images_path
    
    NUM_CLASSES = None
    CLASSES = None
    data_configs = None
    if args.data is not None:
        with open(args.data) as file:
            data_configs = yaml.safe_load(file)
        NUM_CLASSES = data_configs['NC']
        CLASSES = data_configs['CLASSES']
    
    DEVICE = args.device
    OUT_DIR = set_infer_dir(f"{args.name}/{d_set_name}")
    
    model, CLASSES, data_path = load_weights(
        args, DEVICE, DETRModel, data_configs, NUM_CLASSES, CLASSES
    )
    _ = model.to(DEVICE).eval()
    try:
        torchinfo.summary(
            model, 
            device=DEVICE, 
            input_size=(1, 3, args.imgsz, args.imgsz),
            row_settings=["var_names"]
        )
    except:
        print(model)
        # Total parameters and trainable parameters.
        total_params = sum(p.numel() for p in model.parameters())
        print(f"{total_params:,} total parameters.")
        total_trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{total_trainable_params:,} training parameters.")

    # Colors for visualization.
    COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
    DIR_TEST = args.input
    if DIR_TEST == None:
        DIR_TEST = data_path
    test_images = natsorted(collect_all_images(DIR_TEST))
    print(f"Test instances: {len(test_images)}")

    # To count the total number of frames iterated through.
    frame_count = 0
    # To keep adding the frames' FPS.
    total_fps = 0
    
    # For interpretability
    m = model
    target_layers = [#m.model.transformer.decoder.layers[-1],
                    m.model.transformer.decoder.norm,
                    #m.model.class_embed
                    ]
    cam = GradCAM(model=m, target_layers=target_layers, use_cuda=True)
    cam.batch_size = 32
    
    # Create excel variable if test dataset else compute metrics
    if "Test_Dataset" in test_images_path:
        pred_class_pd = pd.DataFrame(columns=["image_ID", "Pred_Class"])
    else:
        m_true_class, m_pred_class = [], []   # Class 0: No bleeding, Class 1: Bleeding
    
    # Loop through the test images
    for image_num in range(len(test_images)):
        test_image_path = test_images[image_num]
        image_name = test_image_path.split(os.path.sep)[-1].split('.')[0]
        orig_image = cv2.imread(test_image_path)
        frame_height, frame_width, _ = orig_image.shape
        if args.imgsz != None:
            RESIZE_TO = args.imgsz
        else:
            RESIZE_TO = frame_width
        
        image_resized = resize(orig_image, RESIZE_TO, square=True)
        image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image = image / 255.0
        image = infer_transforms(image)
        input_tensor = torch.tensor(image, dtype=torch.float32)
        input_tensor = torch.permute(input_tensor, (2, 0, 1))
        input_tensor = input_tensor.unsqueeze(0)
        h, w, _ = orig_image.shape

        start_time = time.time()
        with torch.no_grad():
            outputs = model(input_tensor.to(DEVICE))
        end_time = time.time()
        # Get the current fps.
        fps = 1 / (end_time - start_time)
        # Add `fps` to `total_fps`.
        total_fps += fps
        # Increment frame count.
        frame_count += 1
        
        # Interpretability Plot for each image
        cam_input_tensor = preprocess_image(image, 
                                        mean=[0.5, 0.5, 0.5],
                                        std=[0.5, 0.5, 0.5])
        cam_input_tensor = cam_input_tensor.to(DEVICE)
        grayscale_cam = cam_input_tensor[0, 0, :]
        cam_img = show_cam_on_image(orig_image/255, 
                                    grayscale_cam.cpu(), 
                                    use_rgb=True)
        if not os.path.exists('results/visualizations'):
            os.makedirs('results/visualizations')
        cv2.imwrite(f'results/visualizations/{d_set_name}/viz_{image_name}.jpg', 
                    cam_img)
        
        # Compute Metrics and 
        # Display the predicted class and bounding boxes
        if not "Test_Dataset" in test_image_path:
            if 'non-bleed' in test_image_path:    # 0 for non-bleed, 1 for bleed class
                m_true_class.append(0)
            else:
                m_true_class.append(1)
        
        if len(outputs['pred_boxes'][0]) != 0:
            draw_boxes, pred_classes, scores = convert_detections(
                outputs, 
                args.threshold,
                CLASSES,
                orig_image,
                args 
            )
            
            p_class = 0
            if 'bleeding' in pred_classes:
                p_class = 1
            
            if "Test_Dataset" in test_image_path:
                dict = {
                    "image_ID": image_name,
                    "Pred_Class": "Bleeding" if p_class else "Non-Bleeding" 
                }
                pred_class_pd.loc[len(pred_class_pd)] = dict
            else:
                m_pred_class.append(p_class)

            orig_image = inference_annotations(
                draw_boxes,
                pred_classes,
                scores,
                CLASSES,
                COLORS,
                orig_image,
                args
            )
            if args.show:
                cv2.imshow('Prediction', orig_image)
                cv2.waitKey(1)
        else:
            m_pred_class.append(0)
            
        cv2.imwrite(f"{OUT_DIR}/{image_name}.jpg", orig_image)
        print(f"Image {image_num+1} done...")
        print('-'*50)

    print('TEST PREDICTIONS COMPLETE')
    if args.show:
        cv2.destroyAllWindows()
        # Calculate and print the average FPS.
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    print('')
    
    # Compute Metrics only for the Validation Set
    # Else generate excel sheet for test dataset
    if test_images_path.split('/')[-1].lower() == 'val':
        from sklearn.metrics import accuracy_score, recall_score, f1_score
        
        val_acc = np.around(accuracy_score(m_true_class, m_pred_class), decimals=4)
        val_rec = np.around(recall_score(m_true_class, m_pred_class), decimals=4)
        val_f1 = np.around(f1_score(m_true_class, m_pred_class), decimals=4)
        
        print(f"Validation Set Accuracy: {val_acc}")
        print(f"Validation Set Recall: {val_rec}")
        print(f"Validation Set F1-Score: {val_f1}")
    elif 'Test_Dataset' in test_image_path:
        if not os.path.exists('results/excel'):
            os.makedirs('results/excel')
        pred_class_pd.to_excel(f'results/excel/{d_set_name}.xlsx', index=False)
        
    print('========DONE=============')
        

if __name__ == '__main__':
    args = parse_opt()
    main(args)