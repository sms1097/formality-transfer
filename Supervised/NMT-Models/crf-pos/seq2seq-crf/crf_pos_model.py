import opennmt as onmt


class CrfPosSeq2Seq(onmt.models.SequenceToSequence):
    def __init__(self):
        super().__init__(
            source_inputter = onmt.inputters.ParallelInputter(
                [onmt.inputters.WordEmbedder(embedding_size=512),
                 onmt.inputters.WordEmbedder(embedding_size=64)],
            ),
            target_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
            encoder = onmt.encoders.ParallelEncoder([
                onmt.encoders.LSTMEncoder(
                    num_layers=1,
                    num_units=1024,
                    bidirectional=True
                ), 
                onmt.encoders.LSTMEncoder(
                    num_layers=1,
                    num_units=1024,
                    bidirectional=True
                )
            ], onmt.layers.ConcatReducer(axis=-1)),
            decoder = onmt.decoders.AttentionalRNNDecoder(
                num_layers=2,
                num_units=512
            )
        )


class CrfPosTransformer(onmt.models.Transformer):
    def __init__(self):
        super().__init__(
            source_inputter = onmt.inputters.ParallelInputter(
                [onmt.inputters.WordEmbedder(embedding_size=512),
                 onmt.inputters.WordEmbedder(embedding_size=64)],
                reducer=onmt.layers.ConcatReducer()),
            target_inputter = onmt.inputters.WordEmbedder(embedding_size=512),
            num_layers=6,
            num_units=512,
            num_heads=8,
            ffn_inner_dim=2048,
            dropout=0.1,
            attention_dropout=0.1,
            ffn_dropout=0.1,
            share_encoders=True
        )


model = CrfPosSeq2Seq()
