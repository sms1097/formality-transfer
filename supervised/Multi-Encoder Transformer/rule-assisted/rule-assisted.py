import opennmt as onmt


class DualTransformer(onmt.models.Transformer):
    def __init__(self):
        super().__init__(
            source_inputter = onmt.inputters.ParallelInputter(
                [onmt.inputters.WordEmbedder(embedding_size=512),
                 onmt.inputters.WordEmbedder(embedding_size=512)]),
            target_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1
        )

    def auto_config(self, num_replicas=1):
        config = super().auto_config(num_replicas=num_replicas)
        max_length = config["train"]["maximum_features_length"]
        return onmt.utils.misc.merge_dict(config, {
            "train": {
                "maximum_features_length": [max_length, max_length]
            }
        })


model = DualTransformer()

