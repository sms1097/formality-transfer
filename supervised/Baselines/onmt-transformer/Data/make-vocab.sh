informal_train=src-train.txt
formal_train=trgt-train.txt

informal_target=src-val.txt
formal_target=trgt-val.txt

informal_vocab=src-vocab.txt
formal_vocab=trgt-vocab.txt

onmt-build-vocab --size 12000 --save_vocab $informal_vocab $informal_train
onmt-build-vocab --size 12000 --save_vocab $formal_vocab $formal_vocab