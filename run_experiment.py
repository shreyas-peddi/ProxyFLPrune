import subprocess
import os

# Define variables
n_epochs = 6
use_private_SGD = 1
noise_multiplier = 1.0
l2_norm_clip = 1.0

lr = 0.001
optimizer = 'adam'

major_percent_values = [0.8]  # You can add more values here
dml_weight = 0.5
verbose = 1

# MNIST and Fashion-MNIST configuration
dataset = 'fashion-mnist'  # 'fashion-mnist'
partition_type = 'class'
n_rounds = 100
private_model_type = 'LeNet5'  # LeNet5
proxy_model_type = 'LeNet5'
n_clients = 6
n_client_data = 200
batch_size = 20

# Path configurations
results_dir = "resultsfashionmnistpruneepoch6"

# Create results directory if it doesn't exist
os.makedirs(results_dir, exist_ok=True)

for major_percent in major_percent_values:
    for algorithm in ['ProxyFLPrune','Regular', 'Joint', 'AvgPush', 'FedAvg', 'FML', 'ProxyFL']:
        for sd in range(5):
            # Prepare command for MPI execution
            cmd = [
                'mpiexec', '-n', str(n_clients), 'python', 'run_exp.py',
                f'--dataset={dataset}',
                f'--result_path={results_dir}',
                f'--algorithm={algorithm}',
                f'--seed={sd}',
                f'--partition_type={partition_type}',
                f'--n_clients={n_clients}',
                f'--n_client_data={n_client_data}',
                f'--private_model_type={private_model_type}',
                f'--proxy_model_type={proxy_model_type}',
                f'--use_private_SGD={use_private_SGD}',
                f'--noise_multiplier={noise_multiplier}',
                f'--l2_norm_clip={l2_norm_clip}',
                f'--optimizer={optimizer}',
                f'--lr={lr}',
                f'--dml_weight={dml_weight}',
                f'--major_percent={major_percent}',
                f'--n_epochs={n_epochs}',
                f'--n_rounds={n_rounds}',
                f'--batch_size={batch_size}',
                f'--verbose={verbose}'
            ]
            # Execute the command
            subprocess.run(cmd)

        # Prepare command for plotting results
        cmd_plot = [
            'python', 'plot.py',
            f'--dataset={dataset}',
            f'--result_path={results_dir}',
            '--n_runs=5',
            f'--partition_type={partition_type}',
            f'--n_clients={n_clients}',
            f'--n_client_data={n_client_data}',
            f'--use_private_SGD={use_private_SGD}',
            f'--optimizer={optimizer}',
            f'--lr={lr}',
            f'--noise_multiplier={noise_multiplier}',
            f'--l2_norm_clip={l2_norm_clip}',
            f'--private_model_type={private_model_type}',
            f'--proxy_model_type={proxy_model_type}',
            f'--dml_weight={dml_weight}',
            f'--major_percent={major_percent}',
            f'--batch_size={batch_size}',
            f'--n_epochs={n_epochs}',
            f'--n_rounds={n_rounds}'
        ]
        # Execute the plotting command
        subprocess.run(cmd_plot)
