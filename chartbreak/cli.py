import argparse
import json
from pathlib import Path
from .runner import run
from .analysis import export


def main():
    parser=argparse.ArgumentParser(description='ChartBreak experiment backbone')
    commands=parser.add_subparsers(dest='command',required=True)
    run_parser=commands.add_parser('run',help='Run a small, explicitly bounded experiment')
    run_parser.add_argument('--config',default='configs/demo.json')
    run_parser.add_argument('--limit',type=int,default=5,help='Number of queries per model and strategy (default 5)')
    analyze=commands.add_parser('analyze');analyze.add_argument('run_dir')
    args=parser.parse_args()
    try:
        if args.command=='run':
            print('Saved run:',run(args.config,args.limit))
        else:
            result=export(args.run_dir);print(json.dumps(result['summary'],indent=2))
    except (ValueError,RuntimeError,FileNotFoundError) as error:
        parser.exit(1,str(error)+'\n')
