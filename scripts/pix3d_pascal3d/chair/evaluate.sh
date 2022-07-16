# To use model trained on shapenet train for evaluation on pix3d test
python3 evaluate.py --model=reconstruction --yaml=configs/shapenet/reconstruction_stage_two_pix3d_test.yaml --name=chair --data.pascal3d.cat=chair --load=pretrained_models/shapenet_chair.ckpt --tb= --visdom= --eval.vox_res=128 --eval.icp --group=reconstruction_stage_two --eval_split=val --evaluation --batch_size=1

# To use model trained on pix3d_pascal3d train for evaluation on pix3d test
# output/pix3d_pascal3d/reconstruction_stage_two/chair/checkpoint/ep1200.ckpt
# python3 evaluate.py --model=reconstruction --yaml=configs/pix3d_pascal3d/reconstruction_stage_two.yaml --name=chair --data.pascal3d.cat=chair --load=pretrained_models/pix3d_pascal3d_chair.ckpt --tb= --visdom= --eval.vox_res=128 --eval.icp --group=reconstruction_stage_two --eval_split=val --evaluation --batch_size=1