# deluNet architecture

### Blueprint

* Linear projection of scalar inputs.
* Learned categorical encoding of labelled inputs. 
* Embedding vectors are of dimension 64.
* Prepend [CLS] token to input matrix
* One input becomes a ((F+1) x d) matrix where F is no. of features and d is model dimension.
* A single forward pass will operate on a ((F + 1) x d x B) tensor, where B is batch size.
* Pre-RMSNorm to help with gradient flow.
* Multi head attention to capture feature interrelatedness.
* Standard residual connection with MHA.
* RMSNorm on post-MHA residual connection output.
* SWiGLU activation with hidden dimension of 192.
* Another standard residual connection.
* Classifier FFN with CLS token.
* Softmax over classifier FFN output logits to obtain class probabilities

### Diagramatically:

    20 raw features
        ↓
    feature embeddings
        ↓
    20 × d_model representations
        ↓
    Prepend [CLS]
        ↓

┌─────────────────────────────────┐
│ RMSNorm                         │
│     ↓                           │
│ Multi-Head Attention            │
│     ↓                           │
│ Residual                        │
│     ↓                           │
│ RMSNorm                         │
│     ↓                           │
│ SwiGLU FFN                      │
│     ↓                           │
│ Residual                        │
└─────────────────────────────────┘
             × N blocks
        ↓
    [CLS]
        ↓
    64-dimensional global representation
        ↓
    Linear classification head
        ↓
    C logits
        ↓
    Softmax
        ↓
    class probabilities


### Transformer block

```
X' = X + MHA(RMSNorm(X))
X'' = X' ++ SWiGLU(RMSNorm(X'))
```