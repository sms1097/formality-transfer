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


model = CrfPosSeq2Seq()
