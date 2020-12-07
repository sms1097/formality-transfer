onmt-main --config data.yaml --auto_config infer \
	--features_file Data/src-test.txt Data/pos-train.txt \
	--predictions_file crf_pos_transformer.txt
