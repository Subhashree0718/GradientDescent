Linear Regression with Gradient Descent and Brute Force Search

Overview

This project estimates the best-fit slope for a dataset using two approaches:

Brute-Force Search – Iterates over a range of slope values to find the one that minimizes the absolute error.

Gradient Descent – Optimizes the slope iteratively to minimize the Mean Squared Error (MSE) efficiently.

The project includes data visualization, loss tracking, and a comparison between the two methods.

Features

Generates synthetic linear data with noise.

Implements a brute-force search for the best slope.

Uses gradient descent for efficient slope optimization.

Visualizes:

The generated dataset

Error vs. slope graph (Brute-Force Search)

Loss reduction over iterations (Gradient Descent)

Best-fit line comparisons

Modular object-oriented implementation.

Installation

Ensure you have Python installed. Then, install the required dependencies:

pip install numpy matplotlib

Usage

Run the script using:

python linear_regression.py

This will execute the following steps:

Generate synthetic data.

Perform brute-force search for the best slope.

Perform gradient descent to optimize the slope.

Visualize the results.

Performance Comparison

Brute-Force Search: Exhaustively tests multiple slope values. It is accurate but slow for large datasets.

Gradient Descent: Uses optimization techniques to converge to the best slope efficiently, making it much faster.

Expected Outputs

Scatter Plot of Data: Shows the generated data points with noise.

Error vs. Slope Graph: Displays how the error changes with different slope values in brute-force search.

Loss Curve for Gradient Descent: Illustrates the reduction of loss over iterations.

Comparison of Best-Fit Lines: Shows the best-fit lines from both methods.
