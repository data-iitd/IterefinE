# IterefinE
This repo contains pytorch implementation for the iterative knowledge graph refinement models presented in "IterefinE: Iterative KG Refinement Embeddings using Symbolic Knowledge" at AKBC 2020.

# Running Instructions

After cloning the repository, check if all the above requirements are met. Also, download the datasets if you need.

To run IterefinE, go to combined folder and run the following script-
```
./run_all.sh
```
Here it runs the TypeE-ComplEx and TypeE-ConvE on all the datasets for 6 iterations. To only run TypeE-X models on particular dataset for particular embedding method, use the commands given below-
```
stdbuf -oL ./scripts/wn18rr.sh 6 &> wn18rr.log
stdbuf -oL ./scripts/fb15k-237.sh 6 &> fb15k-237.log
stdbuf -oL ./scripts/yago_new.sh 6 &> yago.log
stdbuf -oL ./scripts/nell_dev.sh 6 &> nell.log
stdbuf -oL ./scripts/fb15k-237_ConvE.sh 6 &> fb15k-237_ConvE.log
stdbuf -oL ./scripts/nell_ConvE_dev.sh 6 &> nell_ConvE.log
stdbuf -oL ./scripts/yago_ConvE_new.sh 6 &> yago_ConvE.log
stdbuf -oL ./scripts/wn18rr_ConvE.sh 6 &> wn18rr_ConvE.log
```

To run the standalonne embedding methods, go to neural folder and run the following script-
```
./train_script.sh
```
Here it runs the ComplEx and ConvE on all the datasets.
