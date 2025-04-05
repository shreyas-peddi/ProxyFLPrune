import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np

import utils

Softmax = nn.Softmax(dim=-1)
LogSoftmax = nn.LogSoftmax(dim=-1)

KL_Loss = nn.KLDivLoss(reduction='batchmean')
CE_Loss = nn.CrossEntropyLoss()


def train_avg(client, eval_data, comm, logger, args):

    proxy_accuracies = np.empty([args.n_clients, args.n_rounds],
                                dtype=np.float32) if comm.rank == 0 else None
    privacy_budgets = np.empty([args.n_clients, args.n_rounds],
                               dtype=np.float32) if comm.rank == 0 else None

    mpi_local_proxy_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_privacy_budgets = np.zeros(args.n_rounds, dtype=np.float32)

    logger.info(f"Hyperparameter setting = {args}")

    start_time = time.time()
    comm_time = 0.

    for r in range(args.n_rounds):

        acc_proxy, _, _ = regular_training_loop(client, None, args)

        proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, args)
        mpi_local_proxy_accuracies[r] = proxy_accuracy
        mpi_local_privacy_budgets[r] = client.privacy_budget

        if args.verbose:
            if args.use_private_SGD:
                logger.info(f"Round {r}, Client {comm.rank}: Proxy acc={proxy_accuracy:.4f} | ε={client.privacy_budget:.2f}")
            else:
                logger.info(f"Round {r}, Client {comm.rank}: Proxy acc={proxy_accuracy:.4f}")

        comm.Barrier()  # for accurate comm time measurement
        comm_start_time = time.time()
        all_model_weights = comm.gather(utils.extract_numpy_weights(client.proxy_model),
                                        root=0)
        comm_time += time.time() - comm_start_time

        avg_weights = utils.extract_numpy_weights(client.proxy_model) if comm.rank == 0 else None
        if comm.rank == 0:
            # Compute average
            for i in range(args.n_clients):
                if i != comm.rank:
                    for k in avg_weights.keys():
                        if not k.endswith("num_batches_tracked"):  # special treatment for BN counting
                            avg_weights[k] += all_model_weights[i][k]
            for k in avg_weights.keys():
                if not k.endswith("num_batches_tracked"):
                    avg_weights[k] /= args.n_clients

        comm.Barrier()  # for accurate comm time measurement
        comm_start_time = time.time()
        avg_weights = comm.bcast(avg_weights, root=0)
        comm_time += time.time() - comm_start_time
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(avg_weights))

    comm.Gather(mpi_local_proxy_accuracies, proxy_accuracies, root=0)
    comm.Gather(mpi_local_privacy_budgets, privacy_budgets, root=0)

    return {"proxy_accuracies": proxy_accuracies,
            "training_time": time.time() - start_time,
            "comm_time": comm_time,
            "privacy_budgets": privacy_budgets,
            }


def train_avg_push(client, eval_data, comm, logger, args):

    proxy_accuracies = np.empty([args.n_clients, args.n_rounds],
                                dtype=np.float32) if comm.rank == 0 else None
    privacy_budgets = np.empty([args.n_clients, args.n_rounds],
                               dtype=np.float32) if comm.rank == 0 else None

    mpi_local_proxy_accuracies = np.zeros(args.n_rounds, dtype=np.float32)

    mpi_local_privacy_budgets = np.zeros(args.n_rounds, dtype=np.float32)

    logger.info(f"Hyperparameter setting = {args}")

    rota = [2 ** i for i in range(int(np.log2(args.n_clients - 1)) + 1)]
    start_time = time.time()
    comm_time = 0.

    for r in range(args.n_rounds):

        acc_proxy, _, _ = regular_training_loop(client, None, args)

        proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, args)
        mpi_local_proxy_accuracies[r] = proxy_accuracy
        mpi_local_privacy_budgets[r] = client.privacy_budget

        if args.verbose:
            if args.use_private_SGD:
                logger.info(f"Round {r}, Client {comm.rank}: Proxy acc={proxy_accuracy:.4f} | ε={client.privacy_budget:.2f}")
            else:
                logger.info(f"Round {r}, Client {comm.rank}: Proxy acc={proxy_accuracy:.4f}")

        source = (comm.rank - rota[r % len(rota)]) % args.n_clients
        dest = (comm.rank + rota[r % len(rota)]) % args.n_clients

        comm.Barrier()  # for accurate comm time measurement
        comm_start_time = time.time()
        new_model_weights = comm.sendrecv(utils.extract_numpy_weights(client.proxy_model),
                                          dest=dest,
                                          source=source)
        comm_time += time.time() - comm_start_time
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(new_model_weights))

    comm.Gather(mpi_local_proxy_accuracies, proxy_accuracies, root=0)
    comm.Gather(mpi_local_privacy_budgets, privacy_budgets, root=0)

    return {"proxy_accuracies": proxy_accuracies,
            "training_time": time.time() - start_time,
            "comm_time": comm_time,
            "privacy_budgets": privacy_budgets,
            }


def train_fml(client, eval_data, comm, logger, args):

    private_accuracies = np.empty([args.n_clients, args.n_rounds],
                                  dtype=np.float32) if comm.rank == 0 else None
    proxy_accuracies = np.empty([args.n_clients, args.n_rounds],
                                dtype=np.float32) if comm.rank == 0 else None
    privacy_budgets = np.empty([args.n_clients, args.n_rounds],
                               dtype=np.float32) if comm.rank == 0 else None

    mpi_local_private_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_proxy_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_privacy_budgets = np.zeros(args.n_rounds, dtype=np.float32)

    logger.info(f"Hyperparameter setting = {args}")

    start_time = time.time()
    comm_time = 0.

    for r in range(args.n_rounds):

        dml_training_loop(client, None, args)

        private_accuracy = utils.evaluate_model(client.private_model, eval_data, args)
        proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, args)
        mpi_local_private_accuracies[r] = private_accuracy
        mpi_local_proxy_accuracies[r] = proxy_accuracy
        mpi_local_privacy_budgets[r] = client.privacy_budget

        if args.verbose:
            if args.use_private_SGD:
                logger.info(f"Round {r}, Client {comm.rank}:" +
                            f" Private acc={private_accuracy:.4f} |" +
                            f" Proxy acc={proxy_accuracy:.4f} |" +
                            f" ε={client.privacy_budget:.2f}")
            else:
                logger.info(f"Round {r}, Client {comm.rank}:" +
                            f" Private acc={private_accuracy:.4f} |" +
                            f" Proxy acc={proxy_accuracy:.4f}")
        comm.Barrier()  # for accurate comm time measurement
        comm_start_time = time.time()
        all_model_weights = comm.gather(utils.extract_numpy_weights(client.proxy_model),
                                        root=0)
        comm_time += time.time() - comm_start_time
        # Compute average
        avg_weights = utils.extract_numpy_weights(client.proxy_model) if comm.rank == 0 else None
        if comm.rank == 0:
            for i in range(args.n_clients):
                if i != comm.rank:
                    for k in avg_weights.keys():
                        if not k.endswith("num_batches_tracked"):
                            avg_weights[k] += all_model_weights[i][k]
            for k in avg_weights.keys():
                if not k.endswith("num_batches_tracked"):
                    avg_weights[k] /= args.n_clients

        comm.Barrier()  # for accurate comm time measurement
        comm_start_time = time.time()
        avg_weights = comm.bcast(avg_weights, root=0)
        comm_time += time.time() - comm_start_time
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(avg_weights))

    comm.Gather(mpi_local_private_accuracies, private_accuracies, root=0)
    comm.Gather(mpi_local_proxy_accuracies, proxy_accuracies, root=0)
    comm.Gather(mpi_local_privacy_budgets, privacy_budgets, root=0)

    return {"private_accuracies": private_accuracies,
            "proxy_accuracies": proxy_accuracies,
            "training_time": time.time() - start_time,
            "comm_time": comm_time,
            "privacy_budgets": privacy_budgets,
            }


def train_proxy_push_prune(client, eval_data, comm, logger, args):
    #args.n_clients: Number of clients participating in the federated learning process.
    #args.n_rounds: Number of communication rounds (training iterations).
    #args.n_epochs: Number of epochs to train the model on each client during a round.
    #args.batch_size: The batch size used for training.
    private_accuracies = np.empty([args.n_clients, args.n_rounds],
                                  dtype=np.float32) if comm.rank == 0 else None
    proxy_accuracies = np.empty([args.n_clients, args.n_rounds],
                                dtype=np.float32) if comm.rank == 0 else None
    privacy_budgets = np.empty([args.n_clients, args.n_rounds],
                               dtype=np.float32) if comm.rank == 0 else None
    #initialization
    mpi_local_private_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_proxy_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_privacy_budgets = np.zeros(args.n_rounds, dtype=np.float32)

    logger.info(f"Hyperparameter setting = {args}")

    rota = [2 ** i for i in range(int(np.log2(args.n_clients - 1)) + 1)]

    start_time = time.time()
    comm_time = 0.
    #For each round t = 0,...,T-1
    for r in range(args.n_rounds):
        #For each client k∈K
        dml_training_loop_prune(client, None, args)
        #private_accuracy: Accuracy of the client's private model.
        #eval_data:The dataset used for evaluating the models' performance after each training round
        private_accuracy = utils.evaluate_model(client.private_model, eval_data, args)
        mpi_local_private_accuracies[r] = private_accuracy
        #proxy_accuracy: Accuracy of the client's proxy model.
        proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, args)
        mpi_local_proxy_accuracies[r] = proxy_accuracy
        #client.privacy_budget: The remaining privacy budget after applying differential privacy mechanisms
        mpi_local_privacy_budgets[r] = client.privacy_budget

        source = (comm.rank - rota[r % len(rota)]) % args.n_clients
        dest = (comm.rank + rota[r % len(rota)]) % args.n_clients
        #If args.verbose is enabled, the current round's results, including model accuracies and privacy budgets, are logged for each client
        if args.verbose:
            if args.use_private_SGD:
                logger.info(f"Round {r}, Client {comm.rank}:" +
                            f" Private acc={private_accuracy:.4f}" +
                            f" | Proxy acc={proxy_accuracy:.4f}" +
                            f" | ε={client.privacy_budget:.2f}" +
                            f" | Sending to {dest}, receiving from {source}")
            else:
                logger.info(f"Round {r}, Client {comm.rank}:" +
                            f" Private acc={private_accuracy:.4f}" +
                            f" | Proxy acc={proxy_accuracy:.4f}" +
                            f" | Sending to {dest}, receiving from {source}")

        comm.Barrier()  # Synchronizes all clients at specific points
        comm_start_time = time.time()
        #Exchange model weights with neighbors
        #comm.sendrecv:Sends model weights to one client and receives weights from another.
        new_proxy_weights = comm.sendrecv(utils.extract_numpy_weights(client.proxy_model),
                                          dest=dest,
                                          source=source)
        comm_time += time.time() - comm_start_time
        #Update proxy model
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(new_proxy_weights))
    #Gathers results like accuracies from all clients to the root (rank 0) client.
    comm.Gather(mpi_local_private_accuracies, private_accuracies, root=0)
    comm.Gather(mpi_local_proxy_accuracies, proxy_accuracies, root=0)
    comm.Gather(mpi_local_privacy_budgets, privacy_budgets, root=0)

    return {"private_accuracies": private_accuracies,
            "proxy_accuracies": proxy_accuracies,
            "privacy_budgets": privacy_budgets,
            "training_time": time.time() - start_time,
            "comm_time": comm_time,
            }
    
def train_proxy_push(client, eval_data, comm, logger, args):
    #args.n_clients: Number of clients participating in the federated learning process.
    #args.n_rounds: Number of communication rounds (training iterations).
    #args.n_epochs: Number of epochs to train the model on each client during a round.
    #args.batch_size: The batch size used for training.
    private_accuracies = np.empty([args.n_clients, args.n_rounds],
                                  dtype=np.float32) if comm.rank == 0 else None
    proxy_accuracies = np.empty([args.n_clients, args.n_rounds],
                                dtype=np.float32) if comm.rank == 0 else None
    privacy_budgets = np.empty([args.n_clients, args.n_rounds],
                               dtype=np.float32) if comm.rank == 0 else None
    #initialization
    mpi_local_private_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_proxy_accuracies = np.zeros(args.n_rounds, dtype=np.float32)
    mpi_local_privacy_budgets = np.zeros(args.n_rounds, dtype=np.float32)

    logger.info(f"Hyperparameter setting = {args}")

    rota = [2 ** i for i in range(int(np.log2(args.n_clients - 1)) + 1)]

    start_time = time.time()
    comm_time = 0.
    #For each round t = 0,...,T-1
    for r in range(args.n_rounds):
        #For each client k∈K
        dml_training_loop(client, None, args)
        #private_accuracy: Accuracy of the client's private model.
        #eval_data:The dataset used for evaluating the models' performance after each training round
        private_accuracy = utils.evaluate_model(client.private_model, eval_data, args)
        mpi_local_private_accuracies[r] = private_accuracy
        #proxy_accuracy: Accuracy of the client's proxy model.
        proxy_accuracy = utils.evaluate_model(client.proxy_model, eval_data, args)
        mpi_local_proxy_accuracies[r] = proxy_accuracy
        #client.privacy_budget: The remaining privacy budget after applying differential privacy mechanisms
        mpi_local_privacy_budgets[r] = client.privacy_budget

        source = (comm.rank - rota[r % len(rota)]) % args.n_clients
        dest = (comm.rank + rota[r % len(rota)]) % args.n_clients
        #If args.verbose is enabled, the current round's results, including model accuracies and privacy budgets, are logged for each client
        if args.verbose:
            if args.use_private_SGD:
                logger.info(f"Round {r}, Client {comm.rank}:" +
                            f" Private acc={private_accuracy:.4f}" +
                            f" | Proxy acc={proxy_accuracy:.4f}" +
                            f" | ε={client.privacy_budget:.2f}" +
                            f" | Sending to {dest}, receiving from {source}")
            else:
                logger.info(f"Round {r}, Client {comm.rank}:" +
                            f" Private acc={private_accuracy:.4f}" +
                            f" | Proxy acc={proxy_accuracy:.4f}" +
                            f" | Sending to {dest}, receiving from {source}")

        comm.Barrier()  # Synchronizes all clients at specific points
        comm_start_time = time.time()
        #Exchange model weights with neighbors
        #comm.sendrecv:Sends model weights to one client and receives weights from another.
        new_proxy_weights = comm.sendrecv(utils.extract_numpy_weights(client.proxy_model),
                                          dest=dest,
                                          source=source)
        comm_time += time.time() - comm_start_time
        #Update proxy model
        client.proxy_model.load_state_dict(utils.convert_np_weights_to_tensor(new_proxy_weights))
    #Gathers results like accuracies from all clients to the root (rank 0) client.
    comm.Gather(mpi_local_private_accuracies, private_accuracies, root=0)
    comm.Gather(mpi_local_proxy_accuracies, proxy_accuracies, root=0)
    comm.Gather(mpi_local_privacy_budgets, privacy_budgets, root=0)

    return {"private_accuracies": private_accuracies,
            "proxy_accuracies": proxy_accuracies,
            "privacy_budgets": privacy_budgets,
            "training_time": time.time() - start_time,
            "comm_time": comm_time,
            }


def train_regular(client, eval_data, comm, logger, args):

    # Regular training (with *combined* number of epochs)

    proxy_accuracies = np.empty([args.n_clients, args.n_rounds],
                                dtype=np.float32) if comm.rank == 0 else None
    privacy_budgets = np.empty([args.n_clients, args.n_rounds],
                               dtype=np.float32) if comm.rank == 0 else None

    logger.info(f"Hyperparameter setting = {args}")

    eval_acc = np.zeros(args.n_rounds, dtype=np.float32)
    local_privacy_budget = np.zeros(args.n_rounds, dtype=np.float32)
    start_time = time.time()

    for r in range(args.n_rounds):

        train_acc, _, _ = regular_training_loop(client, None, args)
        eval_acc[r] = utils.evaluate_model(client.proxy_model, eval_data, args)

        local_privacy_budget[r] = client.privacy_budget

        if args.verbose:
            if args.use_private_SGD:
                logger.info(f"Round {r}, Client {comm.rank}: Proxy acc={eval_acc[r]:.4f} | ε={client.privacy_budget:.2f}")
            else:
                logger.info(f"Round {r}, Client {comm.rank}: Proxy acc={eval_acc[r]:.4f}")

    # Collect all results
    comm.Gather(eval_acc, proxy_accuracies, root=0)
    comm.Gather(local_privacy_budget, privacy_budgets, root=0)

    return {"proxy_accuracies": proxy_accuracies,
            "privacy_budgets": privacy_budgets,
            "training_time": time.time() - start_time,
            }

def hybrid_prune(model, unstructured_amount=0.2, structured_amount=0.1):
    """
    Applies hybrid pruning to the given model:
      - Unstructured L1 pruning on Conv2d layers.
      - Structured LN pruning on Linear layers (prunes entire neurons, here along dim=0).
    
    :param model: PyTorch model to prune.
    :param unstructured_amount: Fraction of weights to prune for convolutional layers.
    :param structured_amount: Fraction of neurons (rows) to prune for linear layers.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) and "conv1" not in name:  # Skip first conv layer
            prune.l1_unstructured(module, name="weight", amount=unstructured_amount)
            print(f"Hybrid pruned unstructured {name}.weight with amount {unstructured_amount}")
        elif isinstance(module, nn.Linear) and "fc3" not in name:  # Skip last FC layer
            prune.ln_structured(module, name="weight", amount=structured_amount, n=2, dim=0)
            print(f"Hybrid pruned structured {name}.weight with amount {structured_amount}")
            

def remove_pruning(model):
    """
    Remove pruning reparameterizations from all pruned layers in the model.
    This makes the pruning permanent and ensures the model's parameters have gradients
    that Opacus can access.
    """
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)) and hasattr(module, "weight_orig"):
            prune.remove(module, "weight")
            print(f"Removed pruning reparameterization from {name}")

def dml_training_loop_prune(client, logger, args):
    
    """
    Training loop that performs differential privacy model training along with hybrid pruning.
    
    The client has both a private model and a proxy model. At scheduled epochs, a hybrid pruning
    procedure is applied to both models.
    
    Assumes that:
      - client.private_data is a tuple (data, labels) as numpy arrays.
      - client.private_model and client.proxy_model are already defined and on the correct device.
      - args contains: n_epochs, batch_size, dml_weight, use_private_SGD,
          (optionally) verbose.
    Optional args attributes for hybrid pruning (sample default values provided):
      - hybrid_prune (bool): whether to enable hybrid pruning (default: True)
      - prune_frequency (int): frequency (in epochs) to apply pruning (default: 5)
      - prune_unstructured_amount (float): fraction to prune in Conv layers (default: 0.2)
      - prune_structured_amount (float): fraction to prune in Linear layers (default: 0.1)
     """
    # Set default values if not provided
    if not hasattr(args, "hybrid_prune"):
        args.hybrid_prune = True
    if not hasattr(args, "prune_frequency"):
        args.prune_frequency = 2
    if not hasattr(args, "prune_unstructured_amount"):
        args.prune_unstructured_amount = 0.4
    if not hasattr(args, "prune_structured_amount"):
        args.prune_structured_amount = 0.2
    #Both the private model and proxy model are set to training mode (.train()), allowing them to update their parameters during backpropagation.
    client.private_model.train()
    client.proxy_model.train()
    train_private_acc = []
    train_proxy_acc = []
    train_privacy_budget = []
    #epsilon is initialized to track the differential privacy budget if DP-SGD is enabled.
    epsilon = 0
    #delta is calculated as the inverse of the size of the training dataset. This is the privacy parameter related to the probability that a single individual's data significantly influences the model.
    delta = 1.0 / client.private_data[0].shape[0]

    for e in range(args.n_epochs):

        train_loader = utils.data_loader(args.dataset,
                                         client.private_data[0],
                                         client.private_data[1],
                                         args.batch_size)

        # total_private_loss = 0.0
        correct_private = 0.0
        acc_private = 0.0
        # total_proxy_loss = 0.0
        correct_proxy = 0.0
        acc_proxy = 0.0
        
        #Inside each epoch, the function iterates over mini-batches of data from the client.train_loader
        for idx, (data, target) in enumerate(train_loader):

            client.private_opt.zero_grad()
            client.proxy_opt.zero_grad()

            data = torch.from_numpy(data).to(client.device)
            target = torch.from_numpy(target).to(client.device)
            target=target.long()# only in cifar10
            #Both the private and proxy models make predictions on the current batch of data. The outputs are computed by performing a forward pass through each model
            pred_private = client.private_model(data)
            pred_proxy = client.proxy_model(data)
            #CE: Measures how well the predicted class distribution matches the true labels (target). This is calculated for both the private model (ce_private) and proxy model (ce_proxy).
            #KL: Measures the difference between the output distributions of the private and proxy models. This loss encourages both models to learn from each other.
            ce_private = CE_Loss(pred_private, target)
            kl_private = KL_Loss(LogSoftmax(pred_private), Softmax(pred_proxy.detach()))

            ce_proxy = CE_Loss(pred_proxy, target)
            kl_proxy = KL_Loss(LogSoftmax(pred_proxy), Softmax(pred_private.detach()))
            #The final loss for both models is a weighted combination of cross-entropy loss and KL-divergence loss. The weight args.dml_weight determines the relative importance of the KL term (knowledge sharing).
            loss_private = (1 - args.dml_weight) * ce_private + args.dml_weight * kl_private
            loss_proxy = (1 - args.dml_weight) * ce_proxy + args.dml_weight * kl_proxy
            #After calculating the loss, the gradients with respect to model parameters are computed using backpropagation
            loss_private.backward()
            loss_proxy.backward()
            #After computing the gradients (and applying differential privacy if necessary), the optimizer updates the model parameters.
            client.private_opt.step()
            client.proxy_opt.step()

            # total_private_loss += loss_private
            # avg_private_loss = total_private_loss / (idx + 1)
            pred_private = pred_private.argmax(dim=-1)
            correct_private += pred_private.eq(target.view_as(pred_private)).sum()
            acc_private = correct_private / client.private_data[0].shape[0]
            train_private_acc.append(acc_private.cpu())

            # total_proxy_loss += loss_proxy
            # avg_proxy_loss = total_proxy_loss / (idx + 1)
            pred_proxy = pred_proxy.argmax(dim=-1)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum()
            acc_proxy = correct_proxy / client.private_data[0].shape[0]
            train_proxy_acc.append(acc_proxy.cpu())
            
            if args.use_private_SGD:
                epsilon, optimal_alpha = client.proxy_opt.privacy_engine.get_privacy_spent(delta)
                client.privacy_budget = epsilon
            
            train_privacy_budget.append(epsilon)

            if logger is not None and args.verbose:
                if args.use_private_SGD:
                    logger.info(f"Epoch {e}: private_acc={acc_private:.4f}, proxy_acc={acc_proxy:.4f}, ε={epsilon:.2f} and δ={delta:.4f} at α={optimal_alpha:.2f}")
                else:
                    logger.info(f"Epoch {e}: private_acc={acc_private:.4f}, proxy_acc={acc_proxy:.4f}")

             # At the end of the epoch, apply hybrid pruning if enabled and at the proper frequency
        if hasattr(args, "hybrid_prune") and args.hybrid_prune:
            # Check if current epoch meets the pruning schedule (e.g., every prune_frequency epochs)
            if hasattr(args, "prune_frequency") and (e+1) % args.prune_frequency == 0:
                print(f"Applying hybrid pruning at epoch {e+1}")
                hybrid_prune(client.private_model, 
                             unstructured_amount=args.prune_unstructured_amount, 
                             structured_amount=args.prune_structured_amount)
                hybrid_prune(client.proxy_model, 
                             unstructured_amount=args.prune_unstructured_amount, 
                             structured_amount=args.prune_structured_amount)
                # Optionally, you can remove the pruning reparameterization to make the pruning permanent:
                # for model in [client.private_model, client.proxy_model]:
                #     for name, module in model.named_modules():
                #         if isinstance(module, (nn.Conv2d, nn.Linear)):
                #             prune.remove(module, 'weight')
                # Remove the pruning reparameterization to make the pruning permanent.
                remove_pruning(client.private_model)
                remove_pruning(client.proxy_model)
                
                fine_tune_epochs = 2  # Try fine-tuning for a few epochs
                optimizer = client.private_opt  # You can also try different optimizers
                
                for _ in range(fine_tune_epochs):
                    for data, target in train_loader:
                        optimizer.zero_grad()
                        data, target = data.to(client.device), target.to(client.device)
                        output = client.private_model(data)
                        loss = CE_Loss(output, target)
                        loss.backward()
                        optimizer.step()
    return (np.array(train_private_acc, dtype=np.float32),
            np.array(train_proxy_acc, dtype=np.float32),
            np.array(train_privacy_budget, dtype=np.float32))
    
def dml_training_loop(client, logger, args):
    #Both the private model and proxy model are set to training mode (.train()), allowing them to update their parameters during backpropagation.
    client.private_model.train()
    client.proxy_model.train()
    train_private_acc = []
    train_proxy_acc = []
    train_privacy_budget = []
    #epsilon is initialized to track the differential privacy budget if DP-SGD is enabled.
    epsilon = 0
    #delta is calculated as the inverse of the size of the training dataset. This is the privacy parameter related to the probability that a single individual's data significantly influences the model.
    delta = 1.0 / client.private_data[0].shape[0]

    for e in range(args.n_epochs):

        train_loader = utils.data_loader(args.dataset,
                                         client.private_data[0],
                                         client.private_data[1],
                                         args.batch_size)

        # total_private_loss = 0.0
        correct_private = 0.0
        acc_private = 0.0
        # total_proxy_loss = 0.0
        correct_proxy = 0.0
        acc_proxy = 0.0
        
        #Inside each epoch, the function iterates over mini-batches of data from the client.train_loader
        for idx, (data, target) in enumerate(train_loader):

            client.private_opt.zero_grad()
            client.proxy_opt.zero_grad()

            data = torch.from_numpy(data).to(client.device)
            target = torch.from_numpy(target).to(client.device)
            target=target.long()# only in cifar10
            #Both the private and proxy models make predictions on the current batch of data. The outputs are computed by performing a forward pass through each model
            pred_private = client.private_model(data)
            pred_proxy = client.proxy_model(data)
            #CE: Measures how well the predicted class distribution matches the true labels (target). This is calculated for both the private model (ce_private) and proxy model (ce_proxy).
            #KL: Measures the difference between the output distributions of the private and proxy models. This loss encourages both models to learn from each other.
            ce_private = CE_Loss(pred_private, target)
            kl_private = KL_Loss(LogSoftmax(pred_private), Softmax(pred_proxy.detach()))

            ce_proxy = CE_Loss(pred_proxy, target)
            kl_proxy = KL_Loss(LogSoftmax(pred_proxy), Softmax(pred_private.detach()))
            #The final loss for both models is a weighted combination of cross-entropy loss and KL-divergence loss. The weight args.dml_weight determines the relative importance of the KL term (knowledge sharing).
            loss_private = (1 - args.dml_weight) * ce_private + args.dml_weight * kl_private
            loss_proxy = (1 - args.dml_weight) * ce_proxy + args.dml_weight * kl_proxy
            #After calculating the loss, the gradients with respect to model parameters are computed using backpropagation
            loss_private.backward()
            loss_proxy.backward()
            #After computing the gradients (and applying differential privacy if necessary), the optimizer updates the model parameters.
            client.private_opt.step()
            client.proxy_opt.step()

            # total_private_loss += loss_private
            # avg_private_loss = total_private_loss / (idx + 1)
            pred_private = pred_private.argmax(dim=-1)
            correct_private += pred_private.eq(target.view_as(pred_private)).sum()
            acc_private = correct_private / client.private_data[0].shape[0]
            train_private_acc.append(acc_private.cpu())

            # total_proxy_loss += loss_proxy
            # avg_proxy_loss = total_proxy_loss / (idx + 1)
            pred_proxy = pred_proxy.argmax(dim=-1)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum()
            acc_proxy = correct_proxy / client.private_data[0].shape[0]
            train_proxy_acc.append(acc_proxy.cpu())
            
            if args.use_private_SGD:
                epsilon, optimal_alpha = client.proxy_opt.privacy_engine.get_privacy_spent(delta)
                client.privacy_budget = epsilon
            
            train_privacy_budget.append(epsilon)

            if logger is not None and args.verbose:
                if args.use_private_SGD:
                    logger.info(f"Epoch {e}: private_acc={acc_private:.4f}, proxy_acc={acc_proxy:.4f}, ε={epsilon:.2f} and δ={delta:.4f} at α={optimal_alpha:.2f}")
                else:
                    logger.info(f"Epoch {e}: private_acc={acc_private:.4f}, proxy_acc={acc_proxy:.4f}")

    return (np.array(train_private_acc, dtype=np.float32),
            np.array(train_proxy_acc, dtype=np.float32),
            np.array(train_privacy_budget, dtype=np.float32))


def regular_training_loop(client, logger, args):

    client.proxy_model.train()
    train_proxy_acc = []
    train_privacy_budget = []

    epsilon = 0
    delta = 1.0 / client.private_data[0].shape[0]

    for e in range(args.n_epochs):

        train_loader = utils.data_loader(args.dataset,
                                         client.private_data[0],
                                         client.private_data[1],
                                         args.batch_size)

        # total_proxy_loss = 0.0
        correct_proxy = 0.0
        acc_proxy = 0.0

        for idx, (data, target) in enumerate(train_loader):

            client.proxy_opt.zero_grad()

            data = torch.from_numpy(data).to(client.device)
            target = torch.from_numpy(target).to(client.device)
            target=target.long() #only in cifar10
            pred_proxy = client.proxy_model(data)

            loss_proxy = CE_Loss(pred_proxy, target)
            loss_proxy.backward()

            client.proxy_opt.step()

            # total_proxy_loss += loss_proxy
            # avg_proxy_loss = total_proxy_loss / (idx + 1)
            pred_proxy = pred_proxy.argmax(dim=-1)
            correct_proxy += pred_proxy.eq(target.view_as(pred_proxy)).sum()
            acc_proxy = correct_proxy / client.private_data[0].shape[0]
            train_proxy_acc.append(acc_proxy.cpu())
#The privacy engine computes the current privacy budget (epsilon) and optimal privacy parameter (alpha). This epsilon is stored in client.privacy_budget and appended to the list train_privacy_budget.
            if args.use_private_SGD:
                epsilon, optimal_alpha = client.proxy_opt.privacy_engine.get_privacy_spent(delta)
                client.privacy_budget = epsilon
            
            train_privacy_budget.append(epsilon)

        if logger is not None and args.verbose:
            if args.use_private_SGD:
                logger.info(f"Epoch {e}: train_proxy_acc={acc_proxy:.4f}, ε={epsilon:.2f} and δ={delta:.4f} at α={optimal_alpha:.2f}")
            else:
                logger.info(f"Epoch {e}: train_proxy_acc={acc_proxy:.4f}")

    return (None,
            np.array(train_proxy_acc, dtype=np.float32),
            np.array(train_privacy_budget, dtype=np.float32))


class Trainer(object):

    def __init__(self, args):
        if args.algorithm == 'FML':
            self.train = train_fml
        elif args.algorithm == 'Regular' or args.algorithm == 'Joint':
            self.train = train_regular
        elif args.algorithm == 'ProxyFL':
            self.train = train_proxy_push
        elif args.algorithm == 'ProxyFLPrune':
            self.train = train_proxy_push_prune
        elif args.algorithm == 'FedAvg':
            self.train = train_avg
        elif args.algorithm == 'AvgPush':
            self.train = train_avg_push
        else:
            raise ValueError("Unknown training method")

    def train(self, client, logger, args):
        return self.train(client, logger, args)
