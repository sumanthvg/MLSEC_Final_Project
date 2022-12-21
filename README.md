# MLSEC_Final_Project

To run the project, one must first download the MCS2018 module, then import it with the following snippet:

`import MCS2018_CPU as MCS2018`

Now, while the code itself supprts CUDA, we have also run it on a Ryzen 9 6900 hence, it's primarily meant for being run on a CPU.
Download the blackbox model and move it to the baseline directory.
Now, for some of the essential downloads of the pair data lists and submit list and the student model images, we need to run the following

`python downloader.py --root ./data --main_imgs --student_model_imgs --submit_list --pairs_list`

Follow this up by-
`python prepare_data.py --root data/student_model_imgs/ --datalist_path data/datalist/ --datalist_type train --gpu_id 1;
python prepare_data.py --root data/imgs/ --datalist_path data/datalist/ --datalist_type val --gpu_id 1`
