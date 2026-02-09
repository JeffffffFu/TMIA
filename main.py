from attack.TMIA.breakpoint import attack_breakpoint
from attack.TMIA.transfer import transfer
from training import get_unlearn_method
from parameter_parser import parameter_parser
from attack.baseline.baseline import run_baseline_attack


def main(args):
    if args['attack_method'] == 'None':
        unlearn_method = get_unlearn_method(args['U_method'])
        unlearn_method(args)
    elif args['attack_method'] == 'TMIA':
        attack_breakpoint(args)
    elif args['attack_method'] in ('TW_MIA', 'UW_MIA'):
        run_baseline_attack(args)
    elif args['attack_method'] in ('Transfer_dataset', 'Transfer_model', 'Transfer_unlearning'):
        transfer(args)
    else:
        raise ValueError(f"Unsupported attack_method: {args['attack_method']}")
if __name__ == '__main__':
    args = parameter_parser()
    main(args)
   # retain_eva(args)