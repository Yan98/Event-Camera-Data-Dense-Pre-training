import os
import argparse
import torch
import numpy as np
import cv2
import config as cfg
from Trainer import Model
import glob
from termcolor import colored
import tqdm
torch.backends.cudnn.benchmark = True

def get_image1(x):
    path = x.split("/")
    path, name = path[:-2], path[-1]
    name = name.split("_")    
    name = f"{name[0]}_left.png"
    path += ["image_left"]
    path += [name]
    return "/".join(path)

def get_image2(x):
    path = x.split("/")
    path, name = path[:-2], path[-1]
    name = name.split("_")    
    name = f"{name[1]}_left.png"
    path += ["image_left"]
    path += [name]
    return "/".join(path)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--temp_folder', type=str, default=None)
    parser.add_argument('--inter_num', type=int, default=30,
                        help="interpolate number")
    
    option = parser.parse_args()
    
    views = glob.glob(os.path.join(option.input,"P*"))
    
    TTA = True
    cfg.MODEL_CONFIG['LOGNAME'] = 'ours_t'
    cfg.MODEL_CONFIG['MODEL_ARCH'] = cfg.init_model_config(
        F = 32,
        depth = [2, 2, 2, 4, 4]
    )
    
    model = Model(-1)
    model.load_model()
    model.eval()
    model.device()

    for number_of_view, view in enumerate(reversed(views)):
        option.result_name = os.path.join(view,"event_left.h5")
        option.temp_folder = os.path.join(option.temp_folder, view.split("/")[-3])
        print(f"{option.result_name}")
        save_full_eviz = False
        H = 480 
        W = 640 
        inter_num = option.inter_num
        shape = [1, 3, H, W]
    
        temp = option.temp_folder
        result_name = option.result_name
        if not os.path.exists(temp):
            os.makedirs(temp)
        
        flow12_filenames = sorted(glob.glob(os.path.join(view, "flow", "*_flow.npy")))
        if os.path.exists(result_name):
            print(colored(f"Skip {result_name}"))
            continue
        
        img_temp_path = temp 
        SKIP_THIS = False
        for index in range(len(flow12_filenames) - 1):
            #image1 = get_image1(flow12_filenames[index])
            image2 = get_image2(flow12_filenames[index])
            image2_ = get_image1(flow12_filenames[index+1])
            #image3 = get_image2(flow12_filenames[index+1])
            if image2 != image2_:
                print(colored("Warning! Not continue!"))
                SKIP_THIS = True
                break
        
        if SKIP_THIS:
            continue
        
        out_video = cv2.VideoWriter(f'{img_temp_path}/video.avi',cv2.VideoWriter_fourcc(*'DIVX'), inter_num, (W, H))
        
        for index in tqdm.tqdm(range(len(flow12_filenames) - 1)):
            image1 = get_image1(flow12_filenames[index])
            image2 = get_image2(flow12_filenames[index])
            image2_ = get_image1(flow12_filenames[index+1])
            image3 = get_image2(flow12_filenames[index+1])
            
            result_id = "_".join(flow12_filenames[index].split("/")[-1].split("_")[:2])
            
            start, end = 0.0, 1.0 
    
            tenFirst = torch.FloatTensor(np.ascontiguousarray(cv2.imread(
                image1,cv2.IMREAD_GRAYSCALE)[None, None, :, :].astype(np.float32) * (1.0 / 255.0))).expand(-1,3,-1,-1).cuda()
            tenSecond = torch.FloatTensor(np.ascontiguousarray(cv2.imread(
                image2,cv2.IMREAD_GRAYSCALE)[None, None, :, :].astype(np.float32) * (1.0 / 255.0))).expand(-1,3,-1,-1).cuda()
            tenThird= torch.FloatTensor(np.ascontiguousarray(cv2.imread(
                image3,cv2.IMREAD_GRAYSCALE)[None, None, :, :].astype(np.float32) * (1.0 / 255.0))).expand(-1,3,-1,-1).cuda()
            
            orginal_shape = tenFirst.shape
    
            
            with torch.no_grad():
                preds = model.multi_inference(tenFirst, tenSecond, TTA=TTA, time_list=[i*(1./inter_num) for i in range(inter_num+1)], \
                                              fast_TTA=TTA)
                revese_last = model.multi_inference(tenSecond, tenThird, TTA=TTA, time_list=[0.0], fast_TTA=TTA)[0]
                preds[-1] = model.multi_inference(preds[-1][None, :, :, :], revese_last[None, :, :, :], TTA=TTA, time_list=[0.5], fast_TTA=TTA)[0]
            
            for x in range(inter_num + 1):
                img_out = preds[x]
                img_out = (torch.clip(img_out,0,1).detach().cpu().numpy().transpose(1, 2, 0) * 255.0).astype(np.uint8)
                
                if index != 0 and x == 0:
                    continue

                out_video.write(cv2.cvtColor(img_out, cv2.COLOR_RGB2GRAY)[...,None].repeat(3,-1))
                
        out_video.release()
                    
            
        shot_noise_rate_hz = np.random.uniform(0,0.25)
        down = os.system(
            f"v2e -i {img_temp_path}/video.avi -o {img_temp_path}/v2eoutput --batch_size 16 --overwrite --unique_output_folder false \
                --dvs_h5 events.h5 --dvs_aedat2 None --dvs_text None --no_preview --dvs_exposure duration .033 \
                    --input_slowmotion_factor 1 --slomo_model ./ckpt/SuperSloMo39.ckpt --skip_video_output  \
                        --auto_timestamp_resolution true --pos_thres 0.2 --neg_thres 0.2 --sigma_thres 0.03 \
                            --cutoff_hz 30 --leak_rate_hz 0.1 --shot_noise_rate_hz {shot_noise_rate_hz} --dvs640")
        
        if down == 0:
            os.makedirs("/".join(result_name.split("/")[:-1]),exist_ok=True)
            os.system(f'mv {img_temp_path}/v2eoutput/events.h5 {result_name}')
            print(index, result_name, shot_noise_rate_hz)
                

