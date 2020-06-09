python3 train.py --dataset ../data/neural/yago/ --embed 100 --batch-size 5000 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --file yago_Conve2 --save yago_ConvE_model.pt --early-pred True |& tee yago_ConvE_pred.txt
python3 train.py --dataset ../data/neural/nell/ --embed 200 --batch-size 2500 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --file nell_ConvE2 --save nell_ConvE_model.pt --early-pred True |& tee nell_ConvE_pred.txt
python3 train.py --dataset ../data/neural/wn18rr/ --embed 100 --batch-size 5000 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --file wn18rr_ConvE2 --save wn18rr_ConvE_model.pt --early-pred True |& tee wn18rr_ConvE_pred.txt
python3 train.py --dataset ../data/neural/fb15k-237/ --embed 200 --batch-size 2500 --epochs 10 --cuda 3 --lr 0.01 --neg-samples 2 --model ConvE --file fb15k_ConvE2 --save fb15k_ConvE_model.pt --early-pred True |& tee fb15k_ConvE_pred.txt


# python3 -u train.py --dataset ../data/neural/yago/ --embed 100 --batch-size 5000 --epochs 10 --cuda 4 --lr 0.001 --neg-samples 2 --model ComplEx --file yago_fil_Complex --save yago_ComplEx_model.pt --early-pred True |& tee yago_ComplEx_pred.txt
# python3 -u train.py --dataset ../data/neural/nell/ --embed 100 --batch-size 5000 --epochs 10 --cuda 4 --lr 0.001 --neg-samples 2 --model ComplEx --file nell_fil_Complex --save nell_ComplEx_model.pt --early-pred True |& tee |& tee nell_ComplEx_pred.txt
# python3 -u train.py --dataset ../data/neural/wn18rr/ --embed 100 --batch-size 5000 --epochs 10 --cuda 4 --lr 0.001 --neg-samples 2 --model ComplEx --file wn18rr_fil_Complex --save wn18rr_ComplEx_model.pt --early-pred True |& tee wn18rr_ComplEx_pred.txt
# python3 -u train.py --dataset ../data/neural/yago_Complex/ --embed 200 --batch-size 2500 --epochs 10 --cuda 2 --lr 0.01 --neg-samples 2 --model ConvE --file yago_filtered_ConvE --save yago_ConvE_filtered_model.pt|& tee yago_filtered_ConvE_pred.txt
# python3 -u train.py --dataset ../data/neural/nell_Complex/ --embed 200 --batch-size 2500 --epochs 10 --cuda 2 --lr 0.01 --neg-samples 2 --model ConvE --file nell_filtered_ConvE --save nell_ConvE_filtered_model.pt|& tee nell_filtered_ConvE_pred.txt
# python3 -u train.py --dataset ../data/neural/wn18rr_Complex/ --embed 200 --batch-size 2500 --epochs 10 --cuda 2 --lr 0.01 --neg-samples 2 --model ConvE --file wn18rr_filtered_ConvE --save wn18rr_ConvE_filtered_model.pt|& tee wn18rr_filtered_ConvE_pred.txt

# python3 -u get_filtered_kb.py --dataset ../data/neural/fb15k-237_ComplEx/ --embed 200 --batch-size 2500 --epochs 10 --cuda 2 --lr 0.01 --neg-samples 2 --model ConvE --file fb15k_filtered_ConvE --save fb15k_ConvE_filtered_model.pt

# python3 -u get_filtered_kb.py --dataset ../data/neural/fb15k-237/ --embed 100 --batch-size 5000 --epochs 10 --cuda 4 --lr 0.001 --neg-samples 2 --model ComplEx --file fb15k_fil_Complex --save fb15k_ComplEx_model.pt --early-pred True