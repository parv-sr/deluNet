import numpy as np
from typing import List

class config:
    d_model: int = 64
    n: int = 10
    batch_size: int = 30

    hidden_dim: int = 192

    # Input tensor config: (b x (n + 1) X d)

class Helper:
    def __init__(self) -> None:
        pass

    @staticmethod
    def softmax(x: np.ndarray) -> np.ndarray:
        """
        Applies softmax on a 1-D vector
        Expected shape: (1,)
        """
        for idx, _ in enumerate(x):
            x[idx] = np.exp(x[idx])

        softmax_denom = np.sum(x)

        for idx, _ in enumerate(x):
            x[idx] /= softmax_denom

        return x

    @staticmethod
    def swish(x: np.ndarray) -> np.ndarray:
        """
        Applies swish on a 1-D vector
        Expected shape: (1,)
        """
        for idx, _ in enumerate(x):
            x[idx] = x[idx] / 1 + np.exp(-x[idx])

class RMSNorm:
    def __init__(self, x: np.ndarray) -> None:
        self.x = x
        self.affine_scale_vector: np.ndarray = np.ones(config.d_model)
        self.eps = 1e-5

    def forward(self) -> np.ndarray:
        n = self.x.shape[0]
        d = config.d_model
        output = np.zeros((n, d))
        
        for i in range(n):
            vec_sum = 0.0
            for j in range(d):
                vec_sum += self.x[i, j] ** 2
                
            mean_square = vec_sum / d
            rms = np.sqrt(mean_square + self.eps)
            
            for j in range(d):
                normalized_val = self.x[i, j] / rms
                output[i, j] = normalized_val * self.affine_scale_vector[j]
                
        return output

class SelfAttention:
    def __init__(self, x: np.ndarray, head_dim: int) -> None:
        # Input matrix
        self.x = x
        #Attention matrices
        self.Q: np.ndarray
        self.K: np.ndarray
        self.V: np.ndarray
        # Dimension of each head
        self.head_dim = head_dim

        self._project()

    def _project(self) -> None:
        self.W_Q = np.zeros((self.head_dim, config.d_model))
        self.W_K = np.zeros((self.head_dim, config.d_model))
        self.W_V = np.zeros((self.head_dim, config.d_model))

    def forward(self) -> np.ndarray:
        self.Q = self.x @ self.W_Q
        self.K = self.x @ self.W_K
        self.V = self.x @ self.W_V

        query_key_product = self.Q @ self.K.T
        scaled_query_key_product = query_key_product / np.sqrt(config.d_model)

        n_rows, n_cols = scaled_query_key_product.shape
        H = np.zeros((n_rows, n_cols))

        for i in range(n_rows):
            row_vector = scaled_query_key_product[i, :]
            row_vector = Helper.softmax(row_vector)

            H[i, :] = row_vector

        output = H @ self.V

        return output

            
class MultiHeadAttention:
    def __init__(self, x: np.ndarray, num_heads: int) -> None:
        self.x = x
        self.num_heads = num_heads

        if not config.d_model % num_heads == 0:
            raise ValueError("Number of heads must be divisible by model dimension")

        self.head_dim = config.d_model / num_heads

        self.atention_heads: List[SelfAttention] = []

        for _ in range(num_heads):
            attn_head = SelfAttention(self.x, self.head_dim)
            self.atention_heads.append(attn_head)

    def forward(self) -> np.ndarray:
        n, d = self.x.shape
        head_ouputs: List = []

        for i in range(self.num_heads):
            start_col = i * self.head_dim
            end_col = (i + 1) * self.head_dim

            x_slice = self.x[:, start_col:end_col]

            head_out = self.atention_heads[i].forward(x_slice)

            head_ouputs.append(head_out)

        output = np.concatenate(head_ouputs, axis=1)

        return output


class SwiGLU:
    def __init__(self, x: np.ndarray, hidden_dim: int) -> None:
        self.x = x
        self.hidden_dim = hidden_dim
        self._project()

    def _project(self) -> None:
        self.W_gate = np.zeros((config.d_model, config.hidden_dim))
        self.W_up = np.zeros((config.d_model, config.hidden_dim))

        self.X_gate = self.x @ self.W_gate
        self.X_up = self.x @ self.W_up

    def forward(self) -> np.ndarray:
        pass            

class TransformerBlock:
    def __init__(self) -> None:
        in_matrix: np.ndarray
        