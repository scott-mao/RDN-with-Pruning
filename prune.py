import os
import copy
import torch
import torch.nn.utils.prune as prune
from utils import set_random_seeds, create_model, prepare_dataloader, train_model, save_model, load_model, evaluate_model, create_classification_report,evaluate_model_benchmark
from datasets import TrainDataset, EvalDataset
from models import RDN
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from prettytable import PrettyTable
from ptflops import get_model_complexity_info



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params += param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params


def compute_final_pruning_rate(pruning_rate, num_iterations):
    """A function to compute the final pruning rate for iterative pruning.
        Note that this cannot be applied for global pruning rate if the pruning rate is heterogeneous among different layers.

    Args:
        pruning_rate (float): Pruning rate.
        num_iterations (int): Number of iterations.

    Returns:
        float: Final pruning rate.
    """

    final_pruning_rate = 1 - (1 - pruning_rate)**num_iterations

    return final_pruning_rate


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv2d_use_mask=False,
                            linear_use_mask=False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def iterative_pruning_finetuning(model,
                                 train_dataloader,
                                 train_dataset,
                                 eval_dataloader,
                                 device,
                                 learning_rate,
                                 l1_regularization_strength,
                                 l2_regularization_strength,
                                 learning_rate_decay=0.1,
                                 conv2d_prune_amount=0.4,
                                 linear_prune_amount=0.2,
                                 num_iterations=10,
                                 num_epochs_per_iteration=10,
                                 model_filename_prefix="pruned_model",
                                 model_dir="saved_models",
                                 grouped_pruning=False):

    for i in range(num_iterations):

        print("Pruning and Finetuning {}/{}".format(i + 1, num_iterations))

        print("Pruning...")

        if grouped_pruning == True:
            # Global pruning
            # I would rather call it grouped pruning.
            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=conv2d_prune_amount,
            )
        else:
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=conv2d_prune_amount)
                elif isinstance(module, torch.nn.Linear):
                    prune.l1_unstructured(module,
                                          name="weight",
                                          amount=linear_prune_amount)

        Set5,Set14,BSD100,Urban100 = evaluate_model_benchmark(model=model,device=device)

        print("Test Accuracy Set5 : {:.4f}".format(Set5))
        print("Test Accuracy Set14 : {:.4f}".format(Set14))
        print("Test Accuracy BSD100 : {:.4f}".format(BSD100))
        print("Test Accuracy Urban100 : {:.4f}".format(Urban100))

        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)

        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        print("Fine-tuning...")

        model = train_model(model=model,
                    train_dataloader=train_dataloader,
                    train_dataset=train_dataset,
                    eval_dataloader=eval_dataloader,
                    device=device,
                    l1_regularization_strength=l1_regularization_strength,
                    l2_regularization_strength=l2_regularization_strength,
                    learning_rate=learning_rate * (learning_rate_decay**i),
                    num_epochs=num_epochs_per_iteration)

        Set5, Set14, BSD100, Urban100 = evaluate_model_benchmark(model=model, device=device)

        print("Test Accuracy Set5 : {:.4f}".format(Set5))
        print("Test Accuracy Set14 : {:.4f}".format(Set14))
        print("Test Accuracy BSD100 : {:.4f}".format(BSD100))
        print("Test Accuracy Urban100 : {:.4f}".format(Urban100))



        num_zeros, num_elements, sparsity = measure_global_sparsity(
            model,
            weight=True,
            bias=False,
            conv2d_use_mask=True,
            linear_use_mask=False)



        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

        model_filename = "{}_{}.pth".format(model_filename_prefix, i + 1)
        model_filepath = os.path.join(model_dir, model_filename)
        save_model(model=model,
                   model_dir=model_dir,
                   model_filename=model_filename)
        model = load_model(model=model,
                           model_filepath=model_filepath,
                           device=device)

    return model


def remove_parameters(model):

    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass

    return model


def main():

    num_classes = 10
    random_seed = 1
    l1_regularization_strength = 0
    l2_regularization_strength = 1e-4
    learning_rate = 1e-4
    learning_rate_decay = 1

    cuda_device = torch.device("cuda:0")
    #cpu_device = torch.device("cpu:0")

    model_dir = "BLAH_BLAH"
    model_filename = "rdn_x4.pth"
    model_filename_prefix = "pruned_model"
    pruned_model_filename = "rdn_pruned.pth"
    model_filepath = os.path.join(model_dir, model_filename)
    pruned_model_filepath = os.path.join(model_dir, pruned_model_filename)

    set_random_seeds(random_seed=random_seed)

    # Create an untrained model.
    model = RDN()

    # Load a pretrained model.
    model = load_model(model=model,
                       model_filepath=model_filepath,
                       device=cuda_device)
    count = count_parameters(model)
    print(count)
    ##data loader start##
    train_dataset = TrainDataset("BLAH_BLAH/DIV2K_x4.h5")
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=16,
                                  shuffle=True,
                                  num_workers=8,
                                  pin_memory=True,
                                  drop_last=True)
    eval_dataset = EvalDataset("BLAH_BLAH/Set5_x4.h5")
    eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=1)
    ##data loader start end##

    ##사전 학습된 가중치를 이용하여 가지치기전 첫 번째 결과 확인
    Set5,Set14,BSD100,Urban100 = evaluate_model_benchmark(model=model,device=cuda_device)


    num_zeros, num_elements, sparsity = measure_global_sparsity(model)

    print("Test Accuracy Set5 : {:.4f}".format(Set5))
    print("Test Accuracy Set14 : {:.4f}".format(Set14))
    print("Test Accuracy BSD100 : {:.4f}".format(BSD100))
    print("Test Accuracy Urban100 : {:.4f}".format(Urban100))


    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    print("Iterative Pruning + Fine-Tuning...")

    pruned_model = copy.deepcopy(model)

    # iterative_pruning_finetuning(
    #     model=pruned_model,
    #     train_loader=train_loader,
    #     test_loader=test_loader,
    #     device=cuda_device,
    #     learning_rate=learning_rate,
    #     learning_rate_decay=learning_rate_decay,
    #     l1_regularization_strength=l1_regularization_strength,
    #     l2_regularization_strength=l2_regularization_strength,
    #     conv2d_prune_amount=0.3,
    #     linear_prune_amount=0,
    #     num_iterations=8,
    #     num_epochs_per_iteration=50,
    #     model_filename_prefix=model_filename_prefix,
    #     model_dir=model_dir,
    #     grouped_pruning=True)

    pruned_model=iterative_pruning_finetuning(
        model=pruned_model,
        train_dataloader=train_dataloader,
        train_dataset=train_dataset,
        eval_dataloader=eval_dataloader,
        device=cuda_device,
        learning_rate=learning_rate,
        learning_rate_decay=learning_rate_decay,
        l1_regularization_strength=l1_regularization_strength,
        l2_regularization_strength=l2_regularization_strength,
        conv2d_prune_amount=0.5,
        linear_prune_amount=0,
        num_iterations=1,
        num_epochs_per_iteration=800,
        model_filename_prefix=model_filename_prefix,
        model_dir=model_dir,
        grouped_pruning=True)

    # Apply mask to the parameters and remove the mask.
    print("pruning and fine train end")
    remove_parameters(model=pruned_model)

    Set5,Set14,BSD100,Urban100 = evaluate_model_benchmark(model=pruned_model,device=cuda_device)

    num_zeros, num_elements, sparsity = measure_global_sparsity(pruned_model)

    print("Test Accuracy Set5 : {:.4f}".format(Set5))
    print("Test Accuracy Set14 : {:.4f}".format(Set14))
    print("Test Accuracy BSD100 : {:.4f}".format(BSD100))
    print("Test Accuracy Urban100 : {:.4f}".format(Urban100))

    print("Global Sparsity:")
    print("{:.2f}".format(sparsity))

    save_model(model=pruned_model, model_dir=model_dir, model_filename=model_filename)
    '''
    print("original")
    count = count_parameters(model)
    print(count)
    print("pruning")
    count = count_parameters(pruned_model)
    print(count)
    print("##########before Pruning")
    macs, params = get_model_complexity_info(model, (3, 320, 180), as_strings=True,
                                           print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))

    print("##########after Pruning")
    macs, params = get_model_complexity_info(pruned_model, (3, 320, 180), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    '''
if __name__ == "__main__":

    main()
