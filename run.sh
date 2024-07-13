python train.py --model_name unet --noise 0.0 --experiment unet-0.0
python train.py --model_name q_unet --noise 0.0 --experiment q-unet-0.0

python train.py --model_name unet --noise 0.1 --experiment unet-0.1
python train.py --model_name q_unet --noise 0.1 --experiment q-unet-0.1

python train.py --model_name unet --noise 0.25 --experiment unet-0.25
python train.py --model_name q_unet --noise 0.25 --experiment q-unet-0.25

python train.py --model_name unet --noise 0.5 --experiment unet-0.5
python train.py --model_name q_unet --noise 0.5 --experiment q-unet-0.5

# random_split

python train.py --model_name unet --noise 0.0 --experiment unet-0.0 --random_split 1
python train.py --model_name q_unet --noise 0.0 --experiment q-unet-0.0 --random_split 1

python train.py --model_name unet --noise 0.1 --experiment unet-0.1 --random_split 1
python train.py --model_name q_unet --noise 0.1 --experiment q-unet-0.1 --random_split 1

python train.py --model_name unet --noise 0.25 --experiment unet-0.25 --random_split 1
python train.py --model_name q_unet --noise 0.25 --experiment q-unet-0.25 --random_split 1

python train.py --model_name unet --noise 0.5 --experiment unet-0.5 --random_split 1
python train.py --model_name q_unet --noise 0.5 --experiment q-unet-0.5 --random_split 1

#
# parser.add_argument('-ep', '--epochs', type=int, default=50)
# parser.add_argument('-rs', '--random_split', type=int, default=0)
# parser.add_argument('-n', '--noise', type=float, default=0.0)
# parser.add_argument('-exp', '--experiment', type=str, default='quantum_noise')
# parser.add_argument('-m', '--model_name', type=str, default='unet')
# parser.add_argument('-bs', '--batch_size', type=int, default=32)
# parser.add_argument('--lr', type=float, default=1e-3)
# parser.add_argument('-es', '--early_stop', type=int, default=6)
# parser.add_argument('-rc', '--random_csv', type=str, default='../../qml-data/csv_files/whole_99_org.csv')
# parser.add_argument('-trc', '--train_csv', type=str, default='../../qml-data/csv_files/train_75_org.csv')
# parser.add_argument('-vc', '--val_csv', type=str, default='../../qml-data/csv_files/val_10_org.csv')
# parser.add_argument('-th', '--threshold', type=float, default=0.7)
# parser.add_argument('-dev', '--device', type=str, default='cuda:0')
# parser.add_argument('-fs', '--from_scratch', type=int, default=1)
#
#
