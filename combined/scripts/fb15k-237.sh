#!/bin/bash
mkdir data predictions
cp  ../data/psl_kgi/fb15k-237/* data
# number of episodes is provided as command line argument $1
touch data/Neural.Rel.out

for ((episode=1;episode<=$1;episode++));
do 
	start=`date +%s`
	echo "working with episode"$episode
	echo "Running PSL-KGI"
	java -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.LoadData data/ &> /dev/null # LOAD DATA
	free -g
	free -k
	free -m
	stdbuf -oL java -Xmx60800m -cp ./target/classes/edu/umd/cs/psl/kgi/:./target/classes:`cat classpath.out` edu.umd.cs.psl.kgi.RunKGI &> out # run PSL-KGI
	python3 gen_predicts.py out cat_predicts.txt rel_predicts.txt
	mkdir fb15k-237_$episode
	mv out fb15k-237_$episode/out
	rm -rf out
	mkdir NN_data
	python3 evaluate_and_update.py rel_predicts.txt data/test_relations.txt cat_predicts.txt data/test_labels.txt data/dev_relations.txt data/dev_labels.txt
	python3 generate_types.py cat_predicts.txt data/names.txt data/subclass.txt data/domain.txt data/range.txt 
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
	python3 print_scores.py data/test_relations.txt data/org_test.txt
	mv -f *.txt fb15k-237_$episode/
	mv -f psl.* fb15k-237_$episode/
	mv -f NN_data/ fb15k-237_$episode/NN-data/
	mkdir fb15k-237_$episode/type_complex
	mkdir fb15k-237_$episode/type_complex/model
	mv -f type_complex/*.txt fb15k-237_$episode/type_complex/
	mv -f type_complex/model/model.pt fb15k-237_$episode/type_complex/model/model.pt
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