python3 train.py --dataset ../data/neural/fb15k-237/ --embed 100 --batch-size 5000 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ComplEx --save fb15k_ComplEx_model.pt
python3 train.py --dataset ../data/neural/fb15k-237/ --embed 200 --batch-size 2500 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --save fb15k_ConvE_model.pt
python3 train.py --dataset ../data/neural/nell/ --embed 100 --batch-size 5000 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ComplEx --save nell_ComplEx_model.pt
python3 train.py --dataset ../data/neural/nell/ --embed 200 --batch-size 2500 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --save nell_ConvE_model.pt
python3 train.py --dataset ../data/neural/yago/ --embed 100 --batch-size 5000 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ComplEx --save yago_ComplEx_model.pt
python3 train.py --dataset ../data/neural/yago/ --embed 200 --batch-size 2500 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --save yago_ConvE_model.pt
python3 train.py --dataset ../data/neural/wn18rr/ --embed 100 --batch-size 5000 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ComplEx --save wn18rr_ComplEx_model.pt
python3 train.py --dataset ../data/neural/wn18rr/ --embed 200 --batch-size 2500 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --save wn18rr_ConvE_model.pt