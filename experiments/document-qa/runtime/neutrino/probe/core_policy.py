"""Frozen offline core selection. No execution or online profiling here.

A policy is a measured hypothesis, not an optimality guarantee. Unknown binary
identities use the explicit safe default; malformed or incompatible policies
are rejected. Semantic validity is still enforced by core_codegen.
"""
import hashlib
import json

CANDIDATES=('shift','late','vector','late_vector')

def digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()

def resource_choice(candidates, strategy='resource'):
    """Each row summarizes all calibrated shapes for one compiled function."""
    base=candidates['shift']
    if strategy=='registers':
        key=lambda m:(candidates[m]['registers'],candidates[m]['instructions'],m!='shift',m)
        return min(candidates,key=key)
    if strategy!='resource':raise ValueError(strategy)
    legal=[m for m,r in candidates.items() if r['local_bytes']<=base['local_bytes']
           and r['residency']>=base['residency']]
    key=lambda m:(-candidates[m]['residency'],candidates[m]['local_bytes'],
                  candidates[m]['instructions'],candidates[m]['registers'],m!='shift',m)
    return min(legal,key=key)

def measured_choice(times,resources,minimum_gain=.03):
    """Require all paired calibration rounds to improve and >=3% mean gain.

    Restrict candidates to nonregressing calibrated residency/local memory.
    The threshold is fixed before held-out measurements; it is not a CI.
    """
    base=times['shift'];n=len(base)
    if n<3 or any(len(v)!=n or any(x<=0 for x in v) for v in times.values()):
        raise ValueError('positive paired calibration measurements required')
    good=['shift']
    for mode,values in times.items():
        if mode=='shift':continue
        r=resources[mode];b=resources['shift']
        if (r['local_bytes']<=b['local_bytes'] and r['residency']>=b['residency']
            and all(v<x for v,x in zip(values,base))
            and sum(values)<=(1-minimum_gain)*sum(base)):
            good.append(mode)
    return min(good,key=lambda m:(sum(times[m]),m!='shift',m))

def resolve_policy(policy,kernel,binary_sha,hardware,sources,tools):
    if policy.get('schema')!=1 or policy.get('default')!='shift':
        raise ValueError('unsupported core policy schema/default')
    if policy['hardware_digest']!=digest_json(hardware):raise ValueError('policy hardware mismatch')
    if policy['compiler_sources']!=sources:raise ValueError('policy generator mismatch')
    if policy['compiler_tools']!=tools:raise ValueError('policy compiler mismatch')
    row=policy['functions'].get(kernel)
    reason='selected'
    if row is None:mode,reason='shift','unseen_function'
    elif row['original_sha256']!=binary_sha:mode,reason='shift','changed_binary'
    else:mode=row['mode']
    if mode not in CANDIDATES:raise ValueError('illegal core candidate')
    return dict(mode=mode,reason=reason,kernel=kernel,original_sha256=binary_sha,
                policy_sha256=digest_json(policy))
