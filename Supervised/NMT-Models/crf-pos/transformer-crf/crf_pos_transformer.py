import opennmt as onmt


class CrfPosTransformer(onmt.models.Transformer):
    def __init__(self):
        super().__init__(
            source_inputter = onmt.inputters.ParallelInputter(
                [onmt.inputters.WordEmbedder(embedding_size=512),
                 onmt.inputters.WordEmbedder(embedding_size=512)],
                reducer=onmt.layers.ConcatReducer()),
            target_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1
        )


model = CrfPosTransformer()
