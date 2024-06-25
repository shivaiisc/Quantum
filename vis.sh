python visual.py --model_name unet \
	--model_path ./ckpts/quantum_noise/unet-0.25/best_unet.pth \
	--img_dir ./res/images/ --vis_dir ./res/vis/unet-0.25/

python visual.py --model_name q_unet \
	--model_path ./ckpts/quantum_noise/q_unet-0.25/best_q_unet.pth \
	--img_dir ./res/images/ --vis_dir ./res/vis/q_unet-0.25/

python visual.py --model_name q_unet \
	--model_path ./ckpts/quantum_noise/q_unet-0.1/best_q_unet.pth \
	--img_dir ./res/images/ --vis_dir ./res/vis/q_unet-0.1/

python visual.py --model_name unet \
	--model_path ./ckpts/quantum_noise/unet-0.1/best_unet.pth \
	--img_dir ./res/images/ --vis_dir ./res/vis/unet-0.1/

python visual.py --model_name q_unet \
	--model_path ./ckpts/quantum_noise/q_unet-0.0/best_q_unet.pth \
	--img_dir ./res/images/ --vis_dir ./res/vis/q_unet-0.0/

python visual.py --model_name unet \
	--model_path ./ckpts/quantum_noise/unet-0.0/best_unet.pth \
	--img_dir ./res/images/ --vis_dir ./res/vis/unet-0.0/

./g.sh
