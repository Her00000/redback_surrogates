"""This commandline util shows an example of training a surrogate model
for the Arnett supernova model using PyTorch. It uses a simplified parameter
space to make training feasible in a reasonable time.

The script uses the NormalizedMultilevelSigmoid neural network architecture
defined in redback_surrogates.learned_surrogate_train, which may not be
optimal for this type of surface (but provides an understandable example).

To run the script, use a command like:

    python examples/example_arnett_training.py \
        --batch_size=500 \
        --batch_iters=10 \
        --iters=200 \
        --layers=32,32,64
"""

import argparse

import numpy as np

from astropy.table import Table
from redback.transient_models.supernova_models import arnett_with_features
import torch
import torch.nn as nn

from redback_surrogates.learned_surrogate import LearnedSurrogateModel
from redback_surrogates.learned_surrogate_data import LearnedSurrogateDataset
from redback_surrogates.learned_surrogate_train import (
    add_output_scaling_and_shift,
    evaluate_learned_model,
    NormalizedMultilevelSigmoid,
)

# We scale the output to improve training stability.
_OUTPUT_SCALE = 1e11

def generate_arnett_data(num_samples, output_scale=1.0):
    """Generate training/validation data for the Arnett model with
    randomly sampled parameters and outputs as the spectra.

    :param num_samples: Number of samples to generate.
    :param output_scale: Scale factor to apply to the output grids.
    :return: A LearnedSurrogateDataset with the generated data.
    """
    times = np.array([0.0, 1.0])  # Unused by "spectra" output type

    # Generate the parameter samples. We use a restricted range from
    # the full set of priors provides by redback to simplify training.
    data = {
        "redshift": np.random.uniform(0.1, 1.0, size=num_samples),
        "f_nickel": np.random.uniform(0.1, 1.0, size=num_samples),
        "mej": np.random.uniform(1.0, 20.0, size=num_samples),
        "vej": np.random.uniform(1_000, 5_000, size=num_samples),
        "kappa": np.random.uniform(0.05, 0.25, size=num_samples),
        "kappa_gamma": np.random.uniform(0.02, 0.05, size=num_samples),
        "temperature_floor": np.random.uniform(2000.0, 10000.0, size=num_samples),
    }

    # Compute the output for each sample.
    output_grids = []
    out_times = None
    out_lambdas = None
    for i in range(num_samples):
        grid = arnett_with_features(
            time=times,
            redshift=data["redshift"][i],
            f_nickel=data["f_nickel"][i],
            mej=data["mej"][i],
            vej=data["vej"][i],
            kappa=data["kappa"][i],
            kappa_gamma=data["kappa_gamma"][i],
            temperature_floor=data["temperature_floor"][i],
            output_format="spectra",
        )
        output_grids.append(grid.spectra)

        # Save the time and wavelength grid from the first sample.
        if i == 0:
            out_times = grid.time
            out_lambdas = grid.lambdas

    data["grid"] = np.array(output_grids) * output_scale

    results = LearnedSurrogateDataset(
        Table(data),
        output_column="grid",
        times=out_times,
        wavelengths=out_lambdas,
    )
    return results


def execute(args):
    torch.set_default_dtype(torch.float64)

    # Use a basic training data set to get the input shapes.
    layer_sizes = [int(size) for size in args.layers.split(",")]
    print(f"Training model with layer sizes: {layer_sizes}")
    simple_ds = generate_arnett_data(num_samples=500, output_scale=_OUTPUT_SCALE)
    validation_input = torch.tensor(simple_ds.get_input(), dtype=torch.float64).T
    validation_output = torch.tensor(simple_ds.get_output(), dtype=torch.float64)

    # The min and max values for each parameter come from the predefined ranges used.
    # If you change the parameter ranges in generate_arnett_data, update these accordingly.
    min_vals = torch.tensor([0.1, 0.1, 1.0, 1_000, 0.05, 0.02, 2000.0], dtype=torch.float64)
    max_vals = torch.tensor([1.0, 1.0, 20.0, 5_000, 0.25, 0.05, 10000.0], dtype=torch.float64)

    # Configure the model and training.
    model = NormalizedMultilevelSigmoid(
        input_size=len(simple_ds.parameter_names),
        hidden_sizes=layer_sizes,
        output_shape=validation_output.shape[1:],
        min_vals=min_vals,
        max_vals=max_vals,
    )
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    validation_criterion = nn.MSELoss()  # Mean Squared Error for regression
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for batch in range(args.iterations):
        batch_ds = generate_arnett_data(num_samples=args.batch_size, output_scale=_OUTPUT_SCALE)
        input_vals = torch.tensor(batch_ds.get_input(), dtype=torch.float64).T
        output_vals = torch.tensor(batch_ds.get_output(), dtype=torch.float64)

        if np.max(output_vals.numpy()) > 1.0:
            print("Warning: Output values exceed 1.0 after scaling. Consider increasing output_scale.")
        if np.max(input_vals.numpy()) < 0.01:
            print("Warning: Input values are very small. Consider normalizing inputs.")

        for _ in range(args.batch_iters):
            # Forward pass
            outputs = model(*input_vals)
            loss = criterion(outputs, output_vals)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            last_loss = loss.item()

        # Compute the validation loss. We do this once at the end of every batch.
        vout = model(*validation_input)
        validation_loss = validation_criterion(vout, validation_output)
        print(f"Batch {batch + 1} (scaled) losses: Training={last_loss:.8f}, Validation={validation_loss.item():.8f}")

    # Create the ONNX model with output scaling and shifting.
    input_example = tuple(input_vals[:, 0])
    onnx_program = torch.onnx.export(
        model,
        input_example,
        input_names=simple_ds.parameter_names,
        dynamo=True,
    )
    onnx_model = add_output_scaling_and_shift(
        onnx_program.model_proto,
        scaling_factor=1.0/_OUTPUT_SCALE,
    )

    surrogate_model = LearnedSurrogateModel(
        onnx_model,
        times=simple_ds.times,
        wavelengths=simple_ds.wavelengths,
        param_names=simple_ds.parameter_names,
    )

    # Evaluate the learned model on a test data set. NOTE the scale of this
    # error will be different because it is on UNSCALED outputs.
    test_data = generate_arnett_data(num_samples=1000)
    rmse = evaluate_learned_model(surrogate_model, test_data)
    print(f"Test RMSE (unscaled outputs): {rmse}")


def main():
    parser = argparse.ArgumentParser(
        prog="test_rb_learn_surrogate.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Test Redback surrogate model training",
    )
    parser.add_argument(
        "--layers",
        dest="layers",
        type=str,
        help="Command-separated list of hidden layer sizes for the neural network.",
        required=True,
    )
    parser.add_argument(
        "--iters",
        dest="iterations",
        type=int,
        help="The number of training iterations to perform.",
        required=True,
    )
    parser.add_argument(
        "--batch_size",
        dest="batch_size",
        type=int,
        help="The batch size for training.",
        required=True,
    )
    parser.add_argument(
        "--batch_iters",
        dest="batch_iters",
        type=int,
        help="The number of iterations per batch.",
        required=True,
    )
    args = parser.parse_args()
    execute(args)


if __name__ == "__main__":
    main()
