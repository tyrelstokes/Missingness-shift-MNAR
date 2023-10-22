
import torch
import torch.nn as nn
from torch.nn import Sigmoid, Softplus
from abc import ABC, abstractmethod
from typing import Optional


class DenseDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, device):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Tanh(),
            nn.Linear(input_dim, output_dim),
            nn.Tanh(),
            nn.Linear(output_dim, output_dim)
        )

        self.layers = self.layers.to(device)

    def forward(self, x):
        return self.layers(x)


class FeatureEmbedder(nn.Module):
    """
    Combines each feature value with its feature ID. The embedding of feature IDs is a trainable parameter.

    This is analogous to position encoding in a transformer.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        metadata: Optional[torch.Tensor],
        device: torch.device,
        multiply_weights: bool,
    ):
        """
        Args:
            input_dim (int): Number of features.
            embedding_dim (int): Size of embedding for each feature ID.
            metadata (Optional[torch.Tensor]): Each row represents a feature and each column is a metadata dimension for the feature.
                Shape (input_dim, metadata_dim).
            device (torch.device): Pytorch device to use.
            multiply_weights (bool): Whether or not to take the product of x with embedding weights when feeding through.
        """
        super().__init__()
        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        if metadata is not None:
            assert metadata.shape[0] == input_dim
        self._metadata = metadata
        self._multiply_weights = multiply_weights

        self._embedding_weights = torch.nn.Parameter(
            torch.zeros(input_dim, embedding_dim, device=device), requires_grad=True
        )
        self._embedding_bias = torch.nn.Parameter(torch.zeros(input_dim, 1, device=device), requires_grad=True)
        torch.nn.init.xavier_uniform_(self._embedding_weights)
        torch.nn.init.xavier_uniform_(self._embedding_bias)

    @property
    def output_dim(self) -> int:
        """
        The final output dimension depends on how features and embeddings are combined in the forward method.

        Returns:
            output_dim (int): The final output dimension of the feature embedder.
        """
        metadata_dim = 0 if self._metadata is None else self._metadata.shape[1]
        output_dim = metadata_dim + self._embedding_dim + 2
        return output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Map each element of each set to a vector, according to which feature it represents.

        Args:
            x (torch.Tensor): Data to embed of shape (batch_size, input_dim).

        Returns:
            feature_embedded_x (torch.Tensor): of shape (batch_size * input_dim, output_dim)
        """

        batch_size, _ = x.size()  # Shape (batch_size, input_dim).
        x_flat = x.reshape(batch_size * self._input_dim, 1)

        # Repeat weights and bias for each instance of each feature.
        if self._metadata is not None:
            embedding_weights_and_metadata = torch.cat((self._embedding_weights, self._metadata), dim=1)
            repeat_embedding_weights = embedding_weights_and_metadata.repeat([batch_size, 1, 1])
        else:
            repeat_embedding_weights = self._embedding_weights.repeat([batch_size, 1, 1])

        # Shape (batch_size * input_dim, embedding_dim)
        repeat_embedding_weights = repeat_embedding_weights.reshape([batch_size * self._input_dim, -1])

        repeat_embedding_bias = self._embedding_bias.repeat((batch_size, 1, 1))
        repeat_embedding_bias = repeat_embedding_bias.reshape((batch_size * self._input_dim, 1))

        if self._multiply_weights:
            features_to_concatenate = [
                x_flat,
                x_flat * repeat_embedding_weights,
                repeat_embedding_bias,
            ]
        else:
            features_to_concatenate = [
                x_flat,
                repeat_embedding_weights,
                repeat_embedding_bias,
            ]

        # Shape (batch_size*input_dim, output_dim)
        feature_embedded_x = torch.cat(features_to_concatenate, dim=1)
        return feature_embedded_x

    def __repr__(self):
        return f"FeatureEmbedder(input_dim={self._input_dim}, embedding_dim={self._embedding_dim}, multiply_weights={self._multiply_weights}, output_dim={self.output_dim})"


class SetEncoderBaseModel(ABC, torch.nn.Module):
    """
    Abstract model class.

    This ABC should be inherited by classes that transform a set of observed features to a single set embedding vector.

    To instantiate this class, the forward function has to be implemented.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        set_embedding_dim: int,
        device: torch.device,
    ):
        """
        Args:
            input_dim: Dimension of input data to embedding model.
            embedding_dim: Dimension of embedding for each input.
            set_embedding_dim: Dimension of output set embedding.
            device: torch device to use.
        """
        ABC.__init__(self)
        torch.nn.Module.__init__(self)

        self._input_dim = input_dim
        self._embedding_dim = embedding_dim
        self._set_embedding_dim = set_embedding_dim
        self._device = device

    @abstractmethod
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicting observed variables with shape (batch_size, input_dim). 1 is observed, 0 is un-observed.
        Returns:
            set_embedding: Embedded output tensor with shape (batch_size, set_embedding_dim)
        """
        raise NotImplementedError()


class PointNet(SetEncoderBaseModel):
    """
    Embeds features using a FeatureEmbedder, transforms each feature independently,
    then pools all the features in each set using sum or max.
    """

    feature_embedder_class = FeatureEmbedder

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int,
        set_embedding_dim: int,
        metadata: Optional[torch.Tensor],
        device: torch.device,
        multiply_weights: bool = True,
        encoding_function: str = "sum",
    ):
        """
        Args:
            input_dim: Dimension of input data to embedding model.
            embedding_dim: Dimension of embedding for each input.
            set_embedding_dim: Dimension of output set embedding.
            metadata: Optional torch tensor. Each row represents a feature and each column is a metadata dimension for the feature.
                Shape (input_dim, metadata_dim).
            device: torch device to use.
            multiply_weights: Boolean. Whether or not to take the product of x with embedding weights when feeding
                 through. Defaults to True.
            encoding_function: Function to use to summarise set input. Defaults to "sum".
        """
        super().__init__(input_dim, embedding_dim, set_embedding_dim, device)

        self._feature_embedder = self.feature_embedder_class(
            input_dim, embedding_dim, metadata, device, multiply_weights
        )
        self._set_encoding_func = self._get_function_from_function_name(encoding_function)

        self._forward_sequence = nn.Sequential(
            nn.Linear(self._feature_embedder.output_dim, set_embedding_dim).to(device),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).
            mask: Mask indicating observed variables with shape (batch_size, input_dim). 1 is observed, 0 is unobserved.
        Returns:
            set_embedding: Embedded output tensor with shape (batch_size, set_embedding_dim).
        """
        feature_embedded_x = self._feature_embedder(x)  # Shape (batch_size * input_dim, self._feature_embedder.output_dim)
        batch_size, _ = x.size()
        embedded = self._forward_sequence(feature_embedded_x)
        embedded = embedded.reshape(
            [batch_size, self._input_dim, self._set_embedding_dim]
        )  # Shape (batch_size, input_dim, set_embedding_dim)

        mask = mask.reshape((batch_size, self._input_dim, 1))
        mask = mask.repeat([1, 1, self._set_embedding_dim])  # Shape (batch_size, input_dim, set_embedding_dim)

        masked_embedding = embedded * mask  # Shape (batch_size, input_dim, set_embedding_dim)
        set_embedding = self._set_encoding_func(masked_embedding, dim=1)  # Shape (batch_size, set_embedding_dim)

        return set_embedding

    @staticmethod
    def max_vals(input_tensor: torch.Tensor, dim: int):
        vals, _ = torch.max(input_tensor, dim=dim)
        return vals

    @staticmethod
    def _get_function_from_function_name(name: str):
        if name == "sum":
            return torch.sum
        elif name == "max":
            return PointNet.max_vals

        raise ValueError("Function name should be one of 'sum', 'max'. Was %s." % name)



class MaskNet(torch.nn.Module):
    """
    This implements the Mask net that is used in notmiwae's implementation for self-masking mechanism
    """

    def __init__(self, input_dim: int, device: str):
        """
        Args:
            input_dim: Dimension of observed features.
            device: torch device to use.
        """
        super().__init__()
        self._device = device
        self.__input_dim = input_dim

        self.W = torch.nn.Parameter(torch.zeros([1, input_dim], device=device), requires_grad=True)
        self.b = torch.nn.Parameter(torch.zeros([1, input_dim], device=device), requires_grad=True)

        self.mask_variables = torch.nn.Parameter(torch.zeros(input_dim, device=device), requires_grad=True)

    def forward(self, x):
        """
        Args:
            x: Input tensor with shape (batch_size, input_dim).

        Returns:
            encoded: Encoded output tensor with shape (batch_size, input_dim)
        """  # Run masked values through model.
        output = Sigmoid()(-Softplus()(self.W) * (x - self.b))
        return output
