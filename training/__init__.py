from training.LLM.finetune_LLM import continuous_update_finetune_LLM
from training.unlearn.GA import continuous_update_finetune_GA
from training.unlearn.NPO import continuous_update_finetune_NPO
from training.unlearn.finetune import continuous_update_finetune
from training.unlearn.finetune_mutiple_update import continuous_update_finetune_multiple_update
from training.unlearn.scrub import continuous_update_finetune_scrub
from training.unlearn.finetune_dp import continuous_update_finetune_dp

def get_unlearn_method(name):
    """method usage:

    function(data_loaders, model, criterion, args)"""

    if name == "continuous_update_finetune_NPO":
        return continuous_update_finetune_NPO
    elif name == "continuous_update_finetune_GA":
        return continuous_update_finetune_GA
    elif name == "continuous_update_finetune":
        return continuous_update_finetune
    elif name == "continuous_update_finetune_dp":
        return continuous_update_finetune_dp
    elif name == "continuous_update_finetune_multiple_update":
        return continuous_update_finetune_multiple_update
    elif name == "continuous_update_finetune_LLM":
        return continuous_update_finetune_LLM
    elif name=="continuous_update_finetune_scrub":
        return  continuous_update_finetune_scrub
    else:
        raise NotImplementedError(f"Unlearn method {name} not implemented!")
