### HW1

## Download data from:
https://drive.google.com/drive/folders/1W7BZRpBmd6YOm1TnWVNLJfXpAFORmTEW?usp=sharing

`cd src`

## Make Dataset
`python run_bert.py 
--do_data  
--model 'bert-large-uncased'  
--max_len 256 `

## Train
`python run_bert.py 
--do_train
--model 'bert-large-uncased' 
--cuda 0 
--dir_name 'large_uncased_256_lr1e-5x0.5per1_accum8_fz2_nodeVec_tfidf' 
--max_len 256 
--lr 5e-6 
--lr_step 1 
--gamma 0.5 
--fz 2 
--accum 8 `

## Predict
`python run_bert.py 
--do_test  
--model 'bert-large-uncased' 
--cuda 0 
--dir_name 'large_uncased_256_lr1e-5x0.5per1_accum8_fz2_nodeVec_tfidf'  
--max_len 256 
--checkpoint 2`