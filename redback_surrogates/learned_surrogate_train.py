"""Basic training for LearnedSurrogateModels using a neural network and pytorch."""

import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm

from redback_surrogates.learned_surrogate import LearnedSurrogateModel


class NormalizedMultilevelSigmoid(nn.Module):
    """Wrapper that combines input normalization with the neural network."""

    def __init__(
        self,
        input_size,
        hidden_sizes,
        output_shape,
        min_vals=None,
        max_vals=None,
    ):
        """Initialize the model.

        :param input_size: The number of input parameters.
        :param hidden_sizes: The size(s) of the hidden layers.
        :param output_shape: The shape of the output (e.g., (num_times, num_wavelengths)).
        :param min_vals: Optional tensor of minimum values for each input parameter for normalization.
        :param max_vals: Optional tensor of maximum values for each input parameter for normalization.
        """
        super().__init__()

        # Save the network configuration.
        self.output_shape = output_shape
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        if np.isscalar(hidden_sizes):
            hidden_sizes = [hidden_sizes]

        # If we have range information, compute the scaling factors for normalization.
        if min_vals is not None and max_vals is not None:
            if len(min_vals) != input_size or len(max_vals) != input_size:
                raise ValueError(
                    f"Length of min_vals ({len(min_vals)}) and max_vals ({len(max_vals)}) "
                    f"must match input_size ({input_size})."
                )
            min_vals_tensor = torch.clone(min_vals)
            max_vals_tensor = torch.clone(max_vals)

            # Compute scale for each parameter, avoiding division by zero
            ranges = max_vals_tensor - min_vals_tensor
            ranges = torch.maximum(ranges, torch.tensor(1e-8, dtype=torch.float64))
            scales = 1.0 / ranges
        else:
            # If no range information is provided, use identity normalization.
            min_vals_tensor = torch.zeros(input_size, dtype=torch.float64)
            scales = torch.ones(input_size, dtype=torch.float64)

        self.register_buffer("input_shift", min_vals_tensor)
        self.register_buffer("input_scale", scales)

        # Create the multilevel sigmoid network.
        self.lin_layers = []
        self.sigmoid_layers = []
        prev_size = self.input_size
        for curr_size in hidden_sizes:
            self.lin_layers.append(nn.Linear(prev_size, curr_size))
            self.sigmoid_layers.append(nn.Sigmoid())
            prev_size = curr_size
        self.out_layer = nn.Linear(prev_size, output_shape[0] * output_shape[1])

    def forward(self, *params):
        x = torch.column_stack(params)
        # Do normalization (no-op if no min/max provided)
        x = (x - self.input_shift) * self.input_scale

        # Propagate through each layer of the network
        for lin, sigmoid in zip(self.lin_layers, self.sigmoid_layers):
            x = sigmoid(lin(x))
        x = self.out_layer(x)
        x = x.view(-1, *self.output_shape)
        return x


def train_pytorch_model(
    dataset,
    hidden_sizes=[64, 64],
    training_epochs=100,
    verbose=False,
    scale_output=True,
):
    """Trains a simple neural network surrogate model using PyTorch.

    Parameters
    ----------
    dataset : LearnedSurrogateDataset
        The dataset containing the training data.
    training_epochs : int, optional
        The number of epochs to train the model. Default is 100.
    hidden_sizes : int or list of int, optional
        The size(s) of the hidden layers in the neural network.
        Default is a pair of 64 node layers.
    verbose : bool, optional
        Whether to print training progress. Default is False.
    scale_output : bool, optional
        Whether to scale the output to [0, 1] during training and add
        inverse scaling to the ONNX model. Default is True.

    Returns
    -------
    LearnedSurrogateModel
        The trained surrogate model.
    """
    torch.set_default_dtype(torch.float64)

    # Get the input, its min/max bounds, and the outputs.
    input_vals = torch.tensor(dataset.get_input(), dtype=torch.float64).T
    min_vals = torch.min(input_vals, dim=1).values
    max_vals = torch.max(input_vals, dim=1).values
    output = torch.tensor(dataset.get_output(), dtype=torch.float64)

    # Scale the output to [0, 1] for training if requested
    if scale_output:
        output_min = torch.min(output)
        output_max = torch.max(output)
        output_range = output_max - output_min
        if output_range == 0:
            output_range = torch.tensor(1.0, dtype=torch.float64)
        output_scaled = (output - output_min) / output_range
    else:
        output_scaled = output
        output_min = torch.tensor(0.0, dtype=torch.float64)
        output_range = torch.tensor(1.0, dtype=torch.float64)

    # Configure the model and training.
    model = NormalizedMultilevelSigmoid(
        input_size=len(dataset.parameter_names),
        hidden_sizes=hidden_sizes,
        output_shape=output.shape[1:],
        min_vals=min_vals,
        max_vals=max_vals,
    )
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)

    if verbose:
        print("Starting training...")
    for idx in tqdm(range(training_epochs)):
        # Forward pass
        outputs = model(*input_vals)
        loss = criterion(outputs, output_scaled)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if verbose and (idx + 1) % 5 == 0:
            print(f"Epoch {idx + 1}/{training_epochs}, Loss: {loss.item()}")

    # Create a LearnedSurrogateModel from the trained PyTorch model.
    input_example = tuple(input_vals[:, 0])
    onnx_program = torch.onnx.export(
        model,
        input_example,
        input_names=dataset.parameter_names,
        dynamo=True,
    )
    
    # Add inverse scaling to the ONNX model if output was scaled
    if scale_output:
        onnx_model = add_output_scaling_and_shift(
            onnx_program.model_proto,
            scaling_factor=output_range.item(),
            shift=output_min.item(),
        )
    else:
        onnx_model = onnx_program.model_proto
    
    surrogate_model = LearnedSurrogateModel(
        onnx_model,
        times=dataset.times,
        wavelengths=dataset.wavelengths,
        param_names=dataset.parameter_names,
    )
    return surrogate_model


def add_output_scaling_and_shift(onnx_model, scaling_factor, shift=0.0):
    """Add a scaling and shift operation to the output of an ONNX model.
    
    The transformation applied is: output_final = (output * scaling_factor) + shift
    
    Parameters
    ----------
    onnx_model : onnx.ModelProto
        The ONNX model to modify.
    scaling_factor : float
        The scaling factor to multiply the output by.
    shift : float, optional
        The additive shift to apply after scaling. Default is 0.0.
    
    Returns
    -------
    onnx.ModelProto
        The modified ONNX model with scaling and shift applied to the output.
    """
    try:
        import onnx
        from onnx import helper, numpy_helper
    except ImportError as err:
        raise ImportError(
            "The onnx package is required to modify the ONNX model. "
            "Please install it using 'pip install onnx'."
        ) from err

    # Get the graph
    graph = onnx_model.graph
    
    # Get the original output name and use it to derive intermediate output names
    original_output_name = graph.output[0].name
    unscaled_output_name = original_output_name + "_unscaled"
    scaled_output_name = original_output_name + "_scaled"

    # Rename the model's original output to the intermediate name
    for node in graph.node:
        for i, output_name in enumerate(node.output):
            if output_name == original_output_name:
                node.output[i] = unscaled_output_name

    # Create a constant tensor for the scaling factor
    scale_tensor = numpy_helper.from_array(
        np.array([scaling_factor], dtype=np.float64),
        name="output_scaling_factor"
    )
    graph.initializer.append(scale_tensor)

    # Create a Mul node to multiply the output by the scaling factor
    mul_node = helper.make_node(
        'Mul',
        inputs=[unscaled_output_name, 'output_scaling_factor'],
        outputs=[scaled_output_name],
        name='output_scaling'
    )
    graph.node.append(mul_node)

    # Create a constant tensor for the shift
    shift_tensor = numpy_helper.from_array(
        np.array([shift], dtype=np.float64),
        name="output_shift"
    )
    graph.initializer.append(shift_tensor)

    # Create an Add node to add the shift
    add_node = helper.make_node(
        'Add',
        inputs=[scaled_output_name, 'output_shift'],
        outputs=[original_output_name],
        name='output_shift_add'
    )
    graph.node.append(add_node)

    # Update the graph output to keep the original output name
    graph.output[0].name = original_output_name

    # Run ONNX checker to validate the modified model
    onnx.checker.check_model(onnx_model)

    return onnx_model


def evaluate_learned_model(model, dataset):
    """Evaluates a trained LearnedSurrogateModel model on a given
    LearnedSurrogateDataset.

    Primarily used for computing test set error.

    Parameters
    ----------
    model : LearnedSurrogateModel
        The trained surrogate model.
    dataset : LearnedSurrogateDataset
        The dataset containing the evaluation data.

    Returns
    -------
    float
        The mean squared error of the model on the dataset.
    float
        The max squared error of the model on the dataset.
    """
    individual_mse = []
    individual_maxse = []
    for idx in range(len(dataset)):
        input_params = dataset.get_input_dict(idx)
        true_output = dataset.get_output(idx)

        # Get the model prediction
        predicted_output = model.predict_spectra_grid(**input_params)

        # Compute MSE for this sample
        sq_error = (predicted_output - true_output) ** 2
        mse = np.mean(sq_error)
        maxse = np.max(sq_error)
        individual_mse.append(mse)
        individual_maxse.append(maxse)

    return np.mean(individual_mse), np.max(individual_maxse)
