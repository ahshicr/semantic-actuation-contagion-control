"""Recompute every main-matrix contrast from supplied trial-level inputs."""
import argparse
from pathlib import Path
import json
import os
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE/'code'))
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from analyze_simulation_matrix import summarize

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    assert not args.out.exists(),'Use a fresh output directory'
    args.out.mkdir(parents=True)
    frame=pd.read_csv(HERE/'results/all_trial_analysis_inputs.csv',float_precision='round_trip')
    configs=json.loads((HERE/'frozen_protocol.json').read_text(encoding='utf-8'))['configurations']
    if isinstance(configs,list): configs={item['setting_id']:item['config'] for item in configs}
    plan=json.loads((HERE/'analysis_plan.json').read_text(encoding='utf-8'))
    started=time.perf_counter()
    with threadpool_limits(limits=2):
        report=summarize(frame,configs,plan,args.out)
    for filename in ['all_policy_summaries.csv','all_paired_comparisons.csv']:
        actual=pd.read_csv(args.out/filename,float_precision='round_trip')
        expected=pd.read_csv(HERE/'results'/filename,float_precision='round_trip')
        assert list(actual.columns)==list(expected.columns),filename
        assert actual.shape==expected.shape,filename
        for column in actual:
            if actual[column].dtype.kind in 'fiu':
                assert np.allclose(actual[column],expected[column],rtol=0,atol=1e-12,equal_nan=True),(filename,column)
            else: assert actual[column].equals(expected[column]),(filename,column)
    report.update(status='passed',every_declared_statistic_checked=True,seconds=time.perf_counter()-started,
        absolute_numeric_tolerance=1e-12,network_or_hardware_access=False,new_independent_environments=0)
    (args.out/'reconstruction_check.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report))

if __name__=='__main__': main()
