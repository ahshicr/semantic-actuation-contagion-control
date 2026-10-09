"""Standalone offline CPU replay and statistical reconstruction.

Example from the extracted compact collection:
python joint_action/reconstruct_joint_action.py --data joint_action --out fresh_output
Add --statistics-only to recompute all forty contrasts without trajectories.
No model service, network connection, serial port or GPU is opened.
"""
import argparse
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

OUTCOMES=['mean_actuator_risk','successful_exposure_burden','task_utility_mean',
          'mean_cost_per_node','benign_success_fraction']
POLICIES=['gradient_joint','gradient_network_only','gradient_guard_only']
SCENES=['warehouse','coverage','formation','allocation']

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def reconstruct_statistics(frame,data,out,plan):
    rng=np.random.default_rng(plan['bootstrap_random_state'])
    effects=[]
    for scene in SCENES:
        draws=rng.integers(0,96,size=(2000,96))
        block=frame[frame.scenario==scene]
        for outcome in OUTCOMES:
            wide=block.pivot(index='trial_id',columns='policy',values=outcome).sort_index()
            for baseline in POLICIES[1:]:
                difference=(wide['gradient_joint']-wide[baseline]).to_numpy()
                interval=np.quantile(difference[draws].mean(1),[.025,.975])
                p=1. if np.all(difference==0) else float(wilcoxon(difference,zero_method='pratt',alternative='two-sided',method='auto').pvalue)
                effects.append({'scenario':scene,'outcome':outcome,'method':'gradient_joint','baseline':baseline,
                    'paired_mean_difference':float(difference.mean()),'pointwise_low':float(interval[0]),
                    'pointwise_high':float(interval[1]),'raw_p':p,'negative_count':int((difference<0).sum()),
                    'tie_count':int((difference==0).sum()),'positive_count':int((difference>0).sum()),
                    'environment_count':96})
    effects=pd.DataFrame(effects)
    assert len(effects)==plan['complete_holm_family']==40
    order=np.argsort(effects.raw_p.to_numpy(),kind='stable')
    adjusted=np.empty(40)
    adjusted[order]=np.minimum(1,np.maximum.accumulate(effects.raw_p.to_numpy()[order]*np.arange(40,0,-1)))
    effects['holm_p_all_40']=adjusted
    expected=pd.read_csv(data/'paired_comparisons.csv',float_precision='round_trip')
    for field in effects:
        if effects[field].dtype.kind in 'fiu':
            assert np.allclose(effects[field],expected[field],rtol=0,atol=1e-12),field
        else: assert effects[field].tolist()==expected[field].tolist(),field
    effects.to_csv(out/'paired_comparisons.csv',index=False)
    return {'contrasts':40,'every_effect_interval_and_adjusted_test_reconstructed':True,
            'absolute_numeric_tolerance':1e-12}

def replay(frame,data,out):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import torch
    torch.set_num_threads(2)
    sys.path.insert(0,str(data/'source_snapshot'))
    import sim_core as core
    import run_simulation as simulation
    calibration=json.loads((data/'inputs/semantic_calibration.json').read_text(encoding='utf-8'))
    stock=simulation.controller_action
    stock_joint_calls=0

    def controller(policy,z,adj,types,tables,params,budget_fraction,guard_cost_ratio,return_diagnostics=False):
        nonlocal stock_joint_calls
        started=time.perf_counter()
        assert z.device.type=='cpu'
        sn,sg=core.gradient_benefits(z,adj,types,tables,params,params.horizon)
        if policy=='gradient_network_only': sg=torch.full_like(sg,-torch.inf)
        if policy=='gradient_guard_only': sn=torch.full_like(sn,-torch.inf)
        cn=core.gather(tables['network_cost'],types)
        cg=guard_cost_ratio*core.gather(tables['guard_cost_factor'],types)
        budget=z.new_full((len(z),),z.shape[1]*budget_fraction)
        x,g=core.allocate_discrete(sn,sg,cn,cg,budget)
        if policy=='gradient_joint':
            sx,sguard,_=stock('horizon_gradient_joint',z,adj,types,tables,params,budget_fraction,guard_cost_ratio)
            assert torch.equal(x,sx) and torch.equal(g,sguard)
            stock_joint_calls+=1
        diagnostic={'policy':policy,'forecast_horizon':params.horizon,'same_unmodified_gradient':True,
                    'path_multiplier':False,'available_actions':policy.removeprefix('gradient_')}
        result=(x,g,time.perf_counter()-started)
        return (*result,diagnostic) if return_diagnostics else result

    simulation.controller_action=controller
    generated=[]
    maximum_error=0.
    for scene in SCENES:
        config=json.loads((data/'inputs'/f'{scene}.json').read_text(encoding='utf-8'))
        params=core.Params(**config['params'])
        for first in range(0,96,8):
            ids=list(range(first,first+8))
            env,seeds,environment_hash=simulation.environment(scene,64,ids,120,config['base_seed'],
                config['environment_stream_id'],params,torch.device('cpu'))
            for policy in POLICIES:
                expected=frame[(frame.scenario==scene)&(frame.policy==policy)&frame.trial_id.isin(ids)].sort_values('trial_id')
                assert len(expected)==8 and set(expected.environment_batch_sha256)=={environment_hash}
                assert expected.environment_seed.tolist()==seeds
                rows,calls=simulation.simulate(policy,config,ids,env,seeds,environment_hash,
                    calibration['receiver_type_matrices'],params,calibration=calibration)
                actual=pd.DataFrame(rows).sort_values('trial_id')
                actual['mean_cost_per_node']=actual.mean_cost/64
                actual['benign_success_fraction']=actual.benign_success_mass/actual.benign_request_mass
                for field in actual:
                    if actual[field].dtype.kind in 'fiu' and field in expected:
                        error=float(np.max(np.abs(actual[field].to_numpy()-expected[field].to_numpy())))
                        maximum_error=max(maximum_error,error)
                        assert np.allclose(actual[field],expected[field],rtol=0,atol=1e-9),(scene,first,policy,field,error)
                saved=json.loads((data/'calls'/f'{scene}_{first:03d}_{policy}.json').read_text(encoding='utf-8'))
                assert len(saved)==len(calls)==30
                for recorded,current in zip(saved,calls):
                    for field in ['network_nodes','guard_nodes','receiver_roles','controller_diagnostics']:
                        assert current[field]==recorded[field],(scene,first,policy,field)
                    for field in ['cost','snapshot_ratio','state_observation_rmse','mutation_observation_row_l1_max']:
                        assert np.allclose(current[field],recorded[field],rtol=0,atol=1e-9)
                generated.extend(actual.to_dict('records'))
        print('Verified all 288 CPU trajectories in '+scene,flush=True)
    pd.DataFrame(generated).to_csv(out/'replayed_outcomes.csv',index=False)
    assert len(generated)==1152 and stock_joint_calls==1440
    return {'trajectories':1152,'decision_masks_verified':4320,
            'joint_batch_decisions_equal_to_existing_gradient_rule':stock_joint_calls,
            'all_input_hashes_and_numeric_summaries_verified':True,
            'maximum_numeric_replay_error':maximum_error,'GPU_jobs':0}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--statistics-only',action='store_true')
    args=parser.parse_args()
    data=args.data.resolve(); out=args.out.resolve()
    assert not out.exists(), 'Use a fresh reconstruction directory'
    out.mkdir(parents=True)
    plan=json.loads((data/'analysis_plan.json').read_text(encoding='utf-8'))
    manifest=json.loads((data/'manifest.json').read_text(encoding='utf-8'))
    assert manifest['status']=='completed'
    for rel,expected in plan['inputs'].items(): assert sha(data/rel)==expected,rel
    for filename in ['outcomes.csv','paired_comparisons.csv','policy_summaries.csv']:
        assert sha(data/filename)==manifest['outputs'][filename],filename
    frame=pd.read_csv(data/'outcomes.csv',float_precision='round_trip')
    assert len(frame)==1152 and frame.controller_failures.eq(0).all()
    started=time.perf_counter()
    result={'statistics':reconstruct_statistics(frame,data,out,plan)}
    if not args.statistics_only: result['trajectory_replay']=replay(frame,data,out)
    result.update(status='passed',seconds=time.perf_counter()-started,new_independent_observations=0,
                  network_or_hardware_access=False,source_manifest_sha256=sha(data/'manifest.json'))
    (out/'reconstruction_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result),flush=True)

if __name__=='__main__': main()
