python3 train.py --model=reconstruction --yaml=configs/shapenet/reconstruction_stage_two.yaml --name=car --data.shapenet.cat=car --mean_latent=output/shapenet/reconstruction_stage_one/car/dump/mean_shape_latent.npy --load=output/shapenet/reconstruction_stage_one/car/checkpoint/ep56.ckpt --load_deformnet=output/shapenet/pretrain_deformnet/car/latest.ckpt