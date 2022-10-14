import logging
import os
import random

import numpy as np
import wandb


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.max = -float("inf")
        self.min = float("inf")
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.max = val if val > self.max else self.max
        self.min = val if val < self.min else self.min

    def make_summary(self, key="None"):
        sum_key = key + "/" + "sum"
        count_key = key + "/" + "count"
        avg_key = key + "/" + "avg"
        max_key = key + "/" + "max"
        min_key = key + "/" + "min"
        final_key = key + "/" + "final"
        summary = {
            sum_key: self.sum,
            count_key: self.count,
            avg_key: self.avg,
            max_key: self.max,
            min_key: self.min,
            final_key: self.val,
        }
        return summary




def wandb_log(prefix, sp_values, com_values, update_summary=False, wandb_summary_dict={}):
    """
        prefix + tags.values is the name of sp_values;
        values should include information like:
        {"Acc": 0.9, "Loss":}
        com_values should include information like:
        {"epoch": epoch, }
    """
    new_values = {}
    for k, _ in sp_values.items():
        # new_values[prefix+"/" + k] = sp_values[k]
        new_key = prefix+"/" + k
        new_values[new_key] = sp_values[k]
        if update_summary:
            if new_key not in wandb_summary_dict:
                wandb_summary_dict[new_key] = AverageMeter()
            wandb_summary_dict[new_key].update(new_values[new_key], n=1)
            summary = wandb_summary_dict[new_key].make_summary(new_key)
            for key, valaue in summary.items():
                wandb.run.summary[key] = valaue

    new_values.update(com_values)
    wandb.log(new_values)

class Arguments:
    def __init__(self, cmd_args):
        for arg_key, arg_val in cmd_args.items():
            setattr(self, arg_key, arg_val)


def wandb_init(args):

    log_config = {
        "client_num_in_total": None, 
        "client_num_per_round": args.num_participants, 
        "model": args.model, 
        "dataset": args.dataset,
        "comm_round": args.num_rounds,
        "epochs": args.epochs,
        "federated_optimizer": args.federated_optimizer, 
        "client_optimizer": "sgd",
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size, 
        "frequency_of_the_test": args.frequency_of_the_test,
        "worker_num": args.worker_num,
        "run_name": args.run_name,
        "framework": "Flower",
        "num_loaders": args.num_loaders,
    }

    # args = Arguments(wandb_config)

    if args.enable_wandb:
        wandb_entity = getattr(args, "wandb_entity", None)
        wandb_args = {
            "entity": args.wandb_entity,
            "project": args.wandb_project,
            "name": args.run_name,
            "config": log_config,
        }

        wandb.init(**wandb_args)



# def wandb_init(args):
#     if args.enable_wandb:
#         wandb_only_server = getattr(args, "wandb_only_server", None)
#         if (wandb_only_server and args.rank == 0 and args.process_id == 0) or not wandb_only_server:
#             wandb_entity = getattr(args, "wandb_entity", None)
#             if wandb_entity is not None:
#                 wandb_args = {
#                     "entity": args.wandb_entity,
#                     "project": args.wandb_project,
#                     "config": args,
#                 }
#             else:
#                 wandb_args = {
#                     "project": args.wandb_project,
#                     "config": args,
#                 }

#             if hasattr(args, "run_name"):
#                 wandb_args["name"] = args.run_name

#             if hasattr(args, "wandb_group_id"):
#                 # wandb_args["group"] = args.wandb_group_id
#                 wandb_args["group"] = "Test1"
#                 wandb_args["name"] = f"Client {args.rank}"
#                 wandb_args["job_type"] = str(args.rank)

#             wandb.init(**wandb_args)






