# MLSEC_Final_Project

List of Requirements:
```
numpy
matplotlib
pandas
pyparsing
Pillow
kiwisolver
tqdm
cycler
pdb
scipy
scikit-learn
pytorch (our version ran on 0.3.0)
cycler
python-dateutil
pytz
networkx
OpenCL
```

To run the project, one must first download the MCS2018 module, then import it with the following snippet:

`import MCS2018_CPU as MCS2018`

Now, while the code itself supprts CUDA, we have run it on a Ryzen 9 6900HS and an RTX 3070 GPU.
Parts of the code have been modified for running on OpenCL and not CUDA, due to performance limitations.

Download the blackbox model and move it to the baseline directory.
Now, for some of the essential downloads of the pair data lists and submit list and the student model images, we need to run the following

`python downloader.py --root ./data --main_imgs --student_model_imgs --submit_list --pairs_list`

Follow this up by-

`python prepare_data.py --root data/student_model_imgs/ --datalist_path data/datalist/ --datalist_type train --gpu_id 0;`
`python prepare_data.py --root data/imgs/ --datalist_path data/datalist/ --datalist_type val --gpu_id 0`

This shall prepare the data for the student model.
Following this we can run the following set of commands to train the blackbox student model.
`cd student_net_learning;`
`CUDA_VISIBLE_DEVICES=0 python main.py --name Baseline1 --epochs <decide number here> --batch_size 32 --datalist ../data/datalist_small/ --root ../data/`

Now, to attack the model in an FGSM fashion, we run the follwing:
`cd ..;`
`CUDA_VISIBLE_DEVICES=0 python attacker.py --root ./data/imgs/ --save_root ./baseline1/ --datalist ./data/pairs_list.csv --model_name <Model Choice Goes Here> --checkpoint_path student_net_learning/checkpoint/Baseline1/best_model_ckpt.t7`
