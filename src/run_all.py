"""One-command final experiment reproduction."""
import argparse
from .data_prep import DATA_PATH

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',default=str(DATA_PATH))
    p.add_argument('--stage',choices=['search','cv','test','all'],default='all')
    p.add_argument('--compile-report',action='store_true');args=p.parse_args()
    from .final_experiment import run
    run(args.data,args.stage)
    if args.stage in ['test','all']:
        from .build_report import build
        build(compile_pdf=args.compile_report)
        print('Final results: outputs/final_metrics.csv; notebook: FINAL_Project.ipynb')

if __name__=='__main__':main()
