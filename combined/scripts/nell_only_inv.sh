#!/bin/bash
mkdir data predictions
cp  ../data/psl_kgi/nell_dev/* data
# number of episodes is provided as command line argument $1
rm -rf data/165.onto-wbpg.db.Mut.txt
touch data/165.onto-wbpg.db.Mut.txt
rm -rf data/165.onto-wbpg.db.RMut.txt
touch data/165.onto-wbpg.db.RMut.txt
rm -rf data/165.onto-wbpg.db.Domain.txt
touch data/165.onto-wbpg.db.Domain.txt
rm -rf data/165.onto-wbpg.db.Range2.txt
touch data/165.onto-wbpg.db.Range2.txt
rm -rf data/165.onto-wbpg.db.RSub.txt
touch data/165.onto-wbpg.db.RSub.txt
rm -rf data/NELL.08m.165.cesv.csv.SameEntity.out
touch data/NELL.08m.165.cesv.csv.SameEntity.out
rm -rf data/165.onto-wbpg.db.Sub.txt
touch data/165.onto-wbpg.db.Sub.txt
touch data/Neural.Rel.out

for ((episode=1;episode<=$1;episode++));
do 
	start=`date +%s`
	echo "working with episode"$episode
	echo "Running PSL-KGI"
	java -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.LoadData_nell_org data/ &> /dev/null # LOAD DATA
	free -g
	free -k
	free -m
	stdbuf -oL java -Xmx60800m -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.RunKGI_nell_org &> out # run PSL-KGI
	python3 gen_predicts.py out cat_predicts.txt rel_predicts.txt
	mkdir nell_only_inv_$episode
	mv out nell_only_inv_$episode/out
	rm -rf out
	mkdir NN_data
	python3 evaluate_and_update.py rel_predicts.txt data/test_relations.txt cat_predicts.txt data/test_labels.txt data/dev_relations.txt data/dev_labels.txt
	python3 generate_types.py cat_predicts.txt data/names.txt data/165.onto-wbpg.db.Sub.txt data/165.onto-wbpg.db.Domain.txt data/165.onto-wbpg.db.Range2.txt
	end=`date +%s`
	dt=$(echo "$end - $start" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
	echo "training NN"
	cd type_complex
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	
	free -g
	free -k
	free -m
	nvidia-smi
	cp ../NN_data/valid.txt ../NN_data/valid_backup.txt
	mv ../NN_data/targets.txt ../NN_data/valid.txt # hack to force dump on these traget facts
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	 --dump_from model/model.pt
	cd ..
	cp type_complex/valid_preds.txt predictions/typed_valid_rel_preds.txt
	cp type_complex/test_preds.txt predictions/typed_test_rel_preds.txt
	python3 convert_neural_src.py NN_data/valid.txt type_complex/train_preds.txt data/names.txt data/Neural.Rel.out 
	python3 print_scores.py 
	mv -f *.txt nell_only_inv_$episode/
	mv -f NN_data/ nell_only_inv_$episode/NN-data/
	mkdir nell_only_inv_$episode/type_complex
	mkdir nell_only_inv_$episode/type_complex/model
	mv -f type_complex/*.txt nell_only_inv_$episode/type_complex/
	mv -f type_complex/model/model.pt nell_only_inv_$episode/type_complex/model/model.pt
	rm -rf *.txt
	rm -rf psl.* 
	rm -rf NN_data
	rm -f type_complex/*.txt
	rm -rf type_complex/model/model.pt
	endfin=`date +%s`
	dt=$(echo "$endfin - $end" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
done
# final cleanup
rm -rf data predictions

mkdir data predictions
cp  ../data/psl_kgi/nell_dev/* data
# number of episodes is provided as command line argument $1
rm -rf data/165.onto-wbpg.db.Mut.txt
touch data/165.onto-wbpg.db.Mut.txt
rm -rf data/165.onto-wbpg.db.Inv.txt
touch data/165.onto-wbpg.db.Inv.txt
rm -rf data/165.onto-wbpg.db.Domain.txt
touch data/165.onto-wbpg.db.Domain.txt
rm -rf data/165.onto-wbpg.db.Range2.txt
touch data/165.onto-wbpg.db.Range2.txt
rm -rf data/165.onto-wbpg.db.RSub.txt
touch data/165.onto-wbpg.db.RSub.txt
rm -rf data/NELL.08m.165.cesv.csv.SameEntity.out
touch data/NELL.08m.165.cesv.csv.SameEntity.out
rm -rf data/165.onto-wbpg.db.Sub.txt
touch data/165.onto-wbpg.db.Sub.txt
touch data/Neural.Rel.out

for ((episode=1;episode<=$1;episode++));
do 
	start=`date +%s`
	echo "working with episode"$episode
	echo "Running PSL-KGI"
	java -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.LoadData_nell_org data/ &> /dev/null # LOAD DATA
	free -g
	free -k
	free -m
	stdbuf -oL java -Xmx60800m -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.RunKGI_nell_org &> out # run PSL-KGI
	python3 gen_predicts.py out cat_predicts.txt rel_predicts.txt
	mkdir nell_only_rmut_$episode
	mv out nell_only_rmut_$episode/out
	rm -rf out
	mkdir NN_data
	python3 evaluate_and_update.py rel_predicts.txt data/test_relations.txt cat_predicts.txt data/test_labels.txt data/dev_relations.txt data/dev_labels.txt
	python3 generate_types.py cat_predicts.txt data/names.txt data/165.onto-wbpg.db.Sub.txt data/165.onto-wbpg.db.Domain.txt data/165.onto-wbpg.db.Range2.txt
	end=`date +%s`
	dt=$(echo "$end - $start" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
	echo "training NN"
	cd type_complex
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	
	free -g
	free -k
	free -m
	nvidia-smi
	cp ../NN_data/valid.txt ../NN_data/valid_backup.txt
	mv ../NN_data/targets.txt ../NN_data/valid.txt # hack to force dump on these traget facts
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	 --dump_from model/model.pt
	cd ..
	cp type_complex/valid_preds.txt predictions/typed_valid_rel_preds.txt
	cp type_complex/test_preds.txt predictions/typed_test_rel_preds.txt
	python3 convert_neural_src.py NN_data/valid.txt type_complex/train_preds.txt data/names.txt data/Neural.Rel.out 
	python3 print_scores.py 
	mv -f *.txt nell_only_rmut_$episode/
	mv -f NN_data/ nell_only_rmut_$episode/NN-data/
	mkdir nell_only_rmut_$episode/type_complex
	mkdir nell_only_rmut_$episode/type_complex/model
	mv -f type_complex/*.txt nell_only_rmut_$episode/type_complex/
	mv -f type_complex/model/model.pt nell_only_rmut_$episode/type_complex/model/model.pt
	rm -rf *.txt
	rm -rf psl.* 
	rm -rf NN_data
	rm -f type_complex/*.txt
	rm -rf type_complex/model/model.pt
	endfin=`date +%s`
	dt=$(echo "$endfin - $end" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
done
# final cleanup
rm -rf data predictions

mkdir data predictions
cp  ../data/psl_kgi/nell_dev/* data
# number of episodes is provided as command line argument $1
rm -rf data/165.onto-wbpg.db.Mut.txt
touch data/165.onto-wbpg.db.Mut.txt
rm -rf data/165.onto-wbpg.db.RMut.txt
touch data/165.onto-wbpg.db.RMut.txt
rm -rf data/165.onto-wbpg.db.Domain.txt
touch data/165.onto-wbpg.db.Domain.txt
rm -rf data/165.onto-wbpg.db.Range2.txt
touch data/165.onto-wbpg.db.Range2.txt
rm -rf data/165.onto-wbpg.db.Inv.txt
touch data/165.onto-wbpg.db.Inv.txt
rm -rf data/NELL.08m.165.cesv.csv.SameEntity.out
touch data/NELL.08m.165.cesv.csv.SameEntity.out
rm -rf data/165.onto-wbpg.db.Sub.txt
touch data/165.onto-wbpg.db.Sub.txt
touch data/Neural.Rel.out

for ((episode=1;episode<=$1;episode++));
do 
	start=`date +%s`
	echo "working with episode"$episode
	echo "Running PSL-KGI"
	java -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.LoadData_nell_org data/ &> /dev/null # LOAD DATA
	free -g
	free -k
	free -m
	stdbuf -oL java -Xmx60800m -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.RunKGI_nell_org &> out # run PSL-KGI
	python3 gen_predicts.py out cat_predicts.txt rel_predicts.txt
	mkdir nell_only_rsub_$episode
	mv out nell_only_rsub_$episode/out
	rm -rf out
	mkdir NN_data
	python3 evaluate_and_update.py rel_predicts.txt data/test_relations.txt cat_predicts.txt data/test_labels.txt data/dev_relations.txt data/dev_labels.txt
	python3 generate_types.py cat_predicts.txt data/names.txt data/165.onto-wbpg.db.Sub.txt data/165.onto-wbpg.db.Domain.txt data/165.onto-wbpg.db.Range2.txt
	end=`date +%s`
	dt=$(echo "$end - $start" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
	echo "training NN"
	cd type_complex
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	
	free -g
	free -k
	free -m
	nvidia-smi
	cp ../NN_data/valid.txt ../NN_data/valid_backup.txt
	mv ../NN_data/targets.txt ../NN_data/valid.txt # hack to force dump on these traget facts
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	 --dump_from model/model.pt
	cd ..
	cp type_complex/valid_preds.txt predictions/typed_valid_rel_preds.txt
	cp type_complex/test_preds.txt predictions/typed_test_rel_preds.txt
	python3 convert_neural_src.py NN_data/valid.txt type_complex/train_preds.txt data/names.txt data/Neural.Rel.out 
	python3 print_scores.py 
	mv -f *.txt nell_only_rsub_$episode/
	mv -f NN_data/ nell_only_rsub_$episode/NN-data/
	mkdir nell_only_rsub_$episode/type_complex
	mkdir nell_only_rsub_$episode/type_complex/model
	mv -f type_complex/*.txt nell_only_rsub_$episode/type_complex/
	mv -f type_complex/model/model.pt nell_only_rsub_$episode/type_complex/model/model.pt
	rm -rf *.txt
	rm -rf psl.* 
	rm -rf NN_data
	rm -f type_complex/*.txt
	rm -rf type_complex/model/model.pt
	endfin=`date +%s`
	dt=$(echo "$endfin - $end" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
done
# final cleanup
rm -rf data predictions

mkdir data predictions
cp  ../data/psl_kgi/nell_dev/* data
# number of episodes is provided as command line argument $1
rm -rf data/165.onto-wbpg.db.Mut.txt
touch data/165.onto-wbpg.db.Mut.txt
rm -rf data/165.onto-wbpg.db.RMut.txt
touch data/165.onto-wbpg.db.RMut.txt
rm -rf data/165.onto-wbpg.db.Domain.txt
touch data/165.onto-wbpg.db.Domain.txt
rm -rf data/165.onto-wbpg.db.Range2.txt
touch data/165.onto-wbpg.db.Range2.txt
rm -rf data/165.onto-wbpg.db.Inv.txt
touch data/165.onto-wbpg.db.Inv.txt
# rm -rf data/NELL.08m.165.cesv.csv.SameEntity.out
# touch data/NELL.08m.165.cesv.csv.SameEntity.out
rm -rf data/165.onto-wbpg.db.Sub.txt
touch data/165.onto-wbpg.db.Sub.txt
rm -rf data/165.onto-wbpg.db.RSub.txt
touch data/165.onto-wbpg.db.RSub.txt
touch data/Neural.Rel.out

for ((episode=1;episode<=$1;episode++));
do 
	start=`date +%s`
	echo "working with episode"$episode
	echo "Running PSL-KGI"
	java -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.LoadData_nell_org data/ &> /dev/null # LOAD DATA
	free -g
	free -k
	free -m
	stdbuf -oL java -Xmx60800m -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.RunKGI_nell_org &> out # run PSL-KGI
	python3 gen_predicts.py out cat_predicts.txt rel_predicts.txt
	mkdir nell_only_SAMEENT_$episode
	mv out nell_only_SAMEENT_$episode/out
	rm -rf out
	mkdir NN_data
	python3 evaluate_and_update.py rel_predicts.txt data/test_relations.txt cat_predicts.txt data/test_labels.txt data/dev_relations.txt data/dev_labels.txt
	python3 generate_types.py cat_predicts.txt data/names.txt data/165.onto-wbpg.db.Sub.txt data/165.onto-wbpg.db.Domain.txt data/165.onto-wbpg.db.Range2.txt
	end=`date +%s`
	dt=$(echo "$end - $start" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
	echo "training NN"
	cd type_complex
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	
	free -g
	free -k
	free -m
	nvidia-smi
	cp ../NN_data/valid.txt ../NN_data/valid_backup.txt
	mv ../NN_data/targets.txt ../NN_data/valid.txt # hack to force dump on these traget facts
	python3 main.py -d ../NN_data -m typed_model -a '{"embedding_dim":19, "base_model_name":"complex", "base_model_arguments":{"embedding_dim":180}}' -l softmax_loss -e 20 -r 0.5 -g 0.3 -b 5000 -x 50 -n 50 -v 1 -q 0	 --dump_from model/model.pt
	cd ..
	cp type_complex/valid_preds.txt predictions/typed_valid_rel_preds.txt
	cp type_complex/test_preds.txt predictions/typed_test_rel_preds.txt
	python3 convert_neural_src.py NN_data/valid.txt type_complex/train_preds.txt data/names.txt data/Neural.Rel.out 
	python3 print_scores.py 
	mv -f *.txt nell_only_SAMEENT_$episode/
	mv -f NN_data/ nell_only_SAMEENT_$episode/NN-data/
	mkdir nell_only_SAMEENT_$episode/type_complex
	mkdir nell_only_SAMEENT_$episode/type_complex/model
	mv -f type_complex/*.txt nell_only_SAMEENT_$episode/type_complex/
	mv -f type_complex/model/model.pt nell_only_SAMEENT_$episode/type_complex/model/model.pt
	rm -rf *.txt
	rm -rf psl.* 
	rm -rf NN_data
	rm -f type_complex/*.txt
	rm -rf type_complex/model/model.pt
	endfin=`date +%s`
	dt=$(echo "$endfin - $end" | bc)
	dd=$(echo "$dt/86400" | bc)
	dt2=$(echo "$dt-86400*$dd" | bc)
	dh=$(echo "$dt2/3600" | bc)
	dt3=$(echo "$dt2-3600*$dh" | bc)
	dm=$(echo "$dt3/60" | bc)
	ds=$(echo "$dt3-60*$dm" | bc)
	printf "Total runtime: %d:%02d:%02d:%02.4f\n" $dd $dh $dm $ds
done
# final cleanup
rm -rf data predictions

