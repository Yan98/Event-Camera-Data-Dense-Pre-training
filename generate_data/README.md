# E-TartanAir Data synthesis

1. Download TartanAir dataset by following the [github](https://github.com/castacks/tartanair_tools_), and unzip them. The dataset structure is
```
ROOT
|
--- ENV_NAME_0/                            # environment folder
|       |
|       ---- Easy                          # difficulty level
|       |      |
|       |      ---- P000                   # trajectory folder
|       |      |      |
|       |      |      +--- flow            # 000000_000001_flow/mask.npy - 000xxx_000xxx_flow/mask.npy
|       |      |      +--- image_left      # 000000_left.png - 000xxx_left.png 
|       |      |  
|       |      +--- P001
|       |      .
|       |      .
|       |      |
|       |      +--- P00K
|       |
|       +--- Hard
|
+-- ENV_NAME_1
.
.
|
+-- ENV_NAME_N
```
2. Install [v2e](https://github.com/SensorsINI/v2e), download the [`SuperSloMo39.ckpt`](https://drive.google.com/file/d/1ETID_4xqLpRBrRo1aOT7Yphs3QqWR_fx/view) and [`ours_t.pkl`](https://drive.google.com/drive/folders/16jUa3HkQ85Z5lb5gce1yoaWkP-rdCd0o) checkpoint, keep the checkpoints in `./ckpt/`, and follow [EMA-VFI](https://github.com/MCG-NJU/EMA-VFI) to setup an enviroment.

3. Generate the event by running
```
ROOT=./dataset #assume the dataset has been downloaded in the dataset folder 
for env in $(find $ROOT -mindepth 2 -maxdepth 2 -type d)
do
    python generate_event.py --input $env --temp_folder ./temp
done
```

4. Process the generated event
```
ROOT=./dataset #assume the dataset has been downloaded in the dataset folder 

python3 split_event.py --file_template ${ROOT}/**/event_left.h5
```

5. Store the file path
```
ROOT=/XXX/XXX/dataset #assume the dataset has been downloaded in the dataset folder 

python3 get_file_path.py --dataset_root ${ROOT}
```
 