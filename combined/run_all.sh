stdbuf -oL ./scripts/wn18rr.sh 6 &> wn18rr.log
stdbuf -oL ./scripts/fb15k-237.sh 6 &> fb15k-237.log
stdbuf -oL ./scripts/yago_new.sh 6 &> yago.log
stdbuf -oL ./scripts/nell_dev.sh 6 &> nell.log
stdbuf -oL ./scripts/fb15k-237_ConvE.sh 6 &> fb15k-237_ConvE.log
stdbuf -oL ./scripts/nell_ConvE_dev.sh 6 &> nell_ConvE.log
stdbuf -oL ./scripts/yago_ConvE_new.sh 6 &> yago_ConvE.log
stdbuf -oL ./scripts/wn18rr_ConvE.sh 6 &> wn18rr_ConvE.log