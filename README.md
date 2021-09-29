# Edge-Aware Multi-Scale Progressive Colorization

An re-implementation of a ICASSP 2021 paper "Edge-Aware Multi-Scale Progressive Colorization".

## 1 Training

Please run `run.sh`

The training is ongoing, which lasts approximately 10 days on 8 NVIDIA Titan Xp GPUs. The dataset is full ImageNet (1128167 images) while the batch size equals 64.

Some training samples can be found in the 'samples' folder. (*gt.png is ground truth and *pred.png is the generated results)

The trained model can be found at https://portland-my.sharepoint.com/:f:/g/personal/yzzhao2-c_my_cityu_edu_hk/Em-Fj61Cu81MgtKLHeQ3WUwBwvLoc91U1z5iIClwYwi0gg?e=paSoG9.

## 2 Validation

Please run `val.sh`

## 3 Reference

If you use this code, please cite this webpage. Thanks.
```bash
@misc{zhao2021empcgithub,
  title = {An re-implementation of EMPC},
  howpublished = {\url{https://github.com/zhaoyuzhi/Edge-Aware-Multi-Scale-Progressive-Colorization}},
  note = {Accessed: 2021-09-19}
}
```
