
import torch

from classes.Embedding import SequenceEmbeddingLayer

class Model(torch.nn.Module):
    def __init__(self, embed_format, embed_dim, n_classes=1) -> None:
        super(Model, self).__init__()
        self.seq_embed_layer = SequenceEmbeddingLayer(embed_format, embed_dim)
        self.classifier = torch.nn.Linear(embed_dim, n_classes)


    def forward(self, sequences:list, requires_grad=None):
        # sequences: # [("id1", "MKTV"), ("id2", "KA")]
        outs = []
        for i, (id, seq) in enumerate(sequences):
            seq_rep = self.seq_embed_layer(id, seq, requires_grad) # shape: seq_len, embed_dim
            out = self.classifier(seq_rep) # shape: seq_len, n_classes
            print(out)
            
            outs.append(out)
        return outs


# test cases
seqs = [("id1", "MKTV"),
        ("id2", "KA")]

model = Model(embed_format="esm", embed_dim=768, n_classes=1)
model(seqs)

