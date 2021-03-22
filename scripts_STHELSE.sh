################### IMPORTANT SCRIOPS FOR GROUND-TRUTH SETTING ###################
##### Compositional
python main.py --cfg STHELSE/COM/GT/OURS

##### FEWSHOT
# python main.py --cfg STHELSE/FEWSHOT/GT/OURS/base
# python main.py --cfg STHELSE/FEWSHOT/GT/OURS/5shot
# python main.py --cfg STHELSE/FEWSHOT/GT/OURS/10shot


################### ALL SCRIPTS FOR Something-Else WITH DETECTION SETTING ###################

##### Compositional
### our variants
# python main.py --cfg STHELSE/COM/DET/STIN
# python main.py --cfg STHELSE/COM/DET/STIN16
# python main.py --cfg STHELSE/COM/DET/OURS_BASE
# python main.py --cfg STHELSE/COM/DET/OURS_BASE_APP
# python main.py --cfg STHELSE/COM/DET/OURS_BASE_NONAPP
# python main.py --cfg STHELSE/COM/DET/OURS_SFI
# python main.py --cfg STHELSE/COM/DET/OURS_SFI_GLOBAL
# python main.py --cfg STHELSE/COM/DET/OURS_SFI_PRED --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/COM/DET/OURS_SFI_PRED_GLOBAL --pred_w 8 --ckpt_suffix w8 
# python main.py --cfg STHELSE/COM/DET/OURS_SFI_PRED_GLOBAL16 --pred_w 8 --ckpt_suffix w8 
# python main.py --cfg STHELSE/COM/DET/OURS_SFI_PRED_GLOBAL_PRETRAIN16 --pred_w 8 --ckpt_suffix w8 

### several methods in our instance-centric framework
# python main.py --cfg STHELSE/COM/DET/OURS_NLs
# python main.py --cfg STHELSE/COM/DET/OURS_STRG
# python main.py --cfg STHELSE/COM/DET/OURS_STIN


##### FEWSHOT
### our variants
## (+PAFE)
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_BASE/base
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_BASE/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_BASE/10shot

## (+PAFE+SFI)
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI/base
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI/10shot

## (+PAFE+SFI+SPP)
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED/base
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED/10shot

## I3D+OURS (concatenating global features from I3D)
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL/base --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL/5shot --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL/10shot --pred_w 8 --ckpt_suffix w8

## I3D*+OURS (concatenating global features from I3D pretrained on 16 frames)
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL_PRETRAIN16/base --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL_PRETRAIN16/5shot --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL_PRETRAIN16/10shot --pred_w 8 --ckpt_suffix w8

## (I3D+OURS)* (concatenating global features from I3D, both I3D and our model are trained on 16 frames)
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL16/base --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL16/5shot --pred_w 8 --ckpt_suffix w8
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_SFI_PRED_GLOBAL16/10shot --pred_w 8 --ckpt_suffix w8


### several methods in our instance-centric framework
## NL
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_NLs/base
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_NLs/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_NLs/10shot

## STRG
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_STRG/base
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_STRG/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_STRG/10shot

## STIN
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_STIN/base
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_STIN/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/OURS_STIN/10shot

### STIN from CVPR2020
## I3D+STIN (trained on 8 frames)
# python main.py --cfg STHELSE/FEWSHOT/DET/STIN/base
# python main.py --cfg STHELSE/FEWSHOT/DET/STIN/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/STIN/10shot

## (I3D+STIN)* (trained on 16 frames)
# python main.py --cfg STHELSE/FEWSHOT/DET/STIN16/base #34418
# python main.py --cfg STHELSE/FEWSHOT/DET/STIN16/5shot
# python main.py --cfg STHELSE/FEWSHOT/DET/STIN16/10shot
