"""Physical execution and objective semantics of the independent experiment."""
import json

import numpy as np
import pytest

from continuous_tracking import PROTOCOL
from continuous_tracking.experiment import DEFAULT_CONFIG, build, quality, run_constant
from continuous_tracking.plant import prefix, validate_plan
from continuous_tracking.experiment import Refused, solve_audited
from continuous_tracking.ocp import TrackingPlan


@pytest.fixture(scope='module')
def system():
    return build()


def nominal(ocp):
    m = ocp.m.data
    g = ocp.db.index['G42']
    return dict(x=m['initial_state']['x_h_V_gamma_s'], alpha_prev=m['initial_state']['alpha_rad'],
                t=0., source=g, target=g, fraction=1., rate=0., CL_des=.025)


def test_linear_alpha_prefix_has_no_restart():
    a, t = prefix([.05,.07,.09], [0.,2.,4.], 3.)
    assert np.array_equal(t, [0.,2.,3.])
    assert np.allclose(a, [.05,.07,.08], rtol=0, atol=1e-15)
    assert a[0] == .05


@pytest.mark.parametrize('a,t', [([0,1],[0,0]), ([0,1],[1,2]), ([0,np.nan],[0,1]), ([0],[0])])
def test_invalid_control_plan_rejected(a,t):
    with pytest.raises(ValueError):
        validate_plan(a,t)


@pytest.mark.parametrize('dt', [.1,.5])
def test_constant_alpha_reproduces_original_rhs(system,dt):
    ocp,_,_=system
    c=nominal(ocp)
    nodes=np.full(7,c['alpha_prev'])
    times=np.arange(7)*.5
    new=ocp.plant.integrate(c['x'],nodes,times,c['source'],c['target'],1.,0.,dt)
    old=ocp.plant.base.integrate(c['x'],nodes[1:],times,c['source'],c['target'],1.,0.,dt)
    assert np.array_equal(new['x'],old['x'])
    assert np.array_equal(new['samples'],old['samples'])


def test_rk_stages_use_continuous_alpha(system):
    ocp,_,_=system
    c=nominal(ocp)
    a=.05; rate=.01; dt=.5
    _,ss=ocp.plant.rk(c['x'],a,rate,1.,0.,ocp.db.parameters[c['source']],
                      ocp.db.parameters[c['target']],dt,0.)
    ss=np.asarray(ss)
    for i,at in enumerate([a,a+rate*dt/2,a+rate*dt/2,a+rate*dt,a+rate*dt]):
        expected=ocp.plant.base.quantities(ss[:4,i],at,c['source'],c['target'],1.)
        np.testing.assert_allclose(ss[4:,i],expected,rtol=2e-13,atol=1e-12)


def test_audit_rejects_entry_jump_and_excess_rate(system):
    ocp,_,_=system
    c=nominal(ocp)
    t=np.arange(7)*.5
    jump=np.full(7,c['alpha_prev']+.001)
    audit,_=ocp.plant.audit(c,jump,t,.025)
    assert not audit['valid'] and not audit['continuous_at_entry']
    bad=np.full(7,c['alpha_prev'])
    bad[1]=bad[0]+ocp.m.data['limits']['alpha_rate_radps']*.5*1.1
    audit,_=ocp.plant.audit(c,bad,t,.025)
    assert not audit['valid'] and audit['continuous_at_entry'] and not audit['control_valid']


def test_control_breakpoints_must_match_audit_grid(system):
    ocp,_,_=system
    c=nominal(ocp)
    with pytest.raises(ValueError,match='divide every'):
        ocp.plant.integrate(c['x'],[.05,.05,.05],[0.,.25,3.],c['source'],c['target'],1.,0.,.1)


def test_late_horizon_tracking_and_first_interval_smoothness(system):
    ocp,_,_=system
    c=nominal(ocp)
    dts=ocp.grid(1200)
    b=ocp.build(dts)
    n=len(dts)
    states=np.tile(np.array(c['x'])/ocp.xscale,(n+1,1)).ravel()
    controls=np.full(n,c['alpha_prev']/ocp.uscale)
    p=np.r_[c['x'],c['alpha_prev'],c['CL_des'],1.,0.,
            ocp.db.parameters[c['source']],ocp.db.parameters[c['target']],0.]
    def terms(u):
        return dict(zip(b['term_names'],np.asarray(b['terms'](np.r_[states,u],p)).ravel()))
    base=terms(controls)
    changed=controls.copy()
    changed[15]+=.02/ocp.uscale  # 7.5 s: strictly beyond the old tracking window.
    assert terms(changed)['target'] != pytest.approx(base['target'],abs=1e-6)
    assert terms(changed)['current_tracking_cost'] == base['current_tracking_cost']
    shifted=terms(controls+.001/ocp.uscale)
    assert base['control_rate'] == 0.
    assert shifted['control_rate'] == pytest.approx((.001/ocp.uscale)**2,abs=1e-12)
    # Compiled helpers must evaluate the same objective, constraints and exact
    # derivatives. This checks more than whether two optimizers both converge.
    point=np.r_[states,changed]
    for name in ('primary','secondary'):
        actual=b['primary'] if name=='primary' else b['solver']
        original=b['raw_'+name]
        for fn in ('nlp_f','nlp_g','nlp_grad_f','nlp_jac_g'):
            left=actual.get_function(fn)(point,p)
            right=original.get_function(fn)(point,p)
            left=left if isinstance(left,tuple) else (left,)
            right=right if isinstance(right,tuple) else (right,)
            for l,r in zip(left,right):
                np.testing.assert_array_equal(np.asarray(l),np.asarray(r))


def test_tail_grid_and_quality_flags(system):
    ocp,config,_=system
    assert sum(ocp.grid(1200)) == 60.
    assert sum(ocp.grid(3)) == 3.
    aud={'grids':{'0.1':{'CL_mean_first3':.031,'CL_rmse_first3':.008,'CL_peak_error_first3':.015}}}
    q=quality(aud,.025,config['tracking_quality'])
    assert not q['passed'] and set(q['failed'])=={'mean_abs_error','rmse','peak_abs_error'}
    assert q['status']=='SAFE_WITH_TRACKING_DEVIATION'
    assert not q['deviation_required_for_safety_proven']


def test_protocol_and_existing_output_refusal(tmp_path):
    cfg=json.loads(DEFAULT_CONFIG.read_text(encoding='utf-8'))
    assert cfg['protocol']==PROTOCOL
    cfg['protocol']='legacy'
    path=tmp_path/'wrong.json'
    path.write_text(json.dumps(cfg),encoding='utf-8')
    with pytest.raises(ValueError,match='protocol mismatch'):
        build(path)
    with pytest.raises(FileExistsError):
        run_constant(tmp_path,.025,1)


def test_solver_success_is_not_a_certificate(tmp_path):
    from types import SimpleNamespace
    plan=TrackingPlan(True,'Solve_Succeeded',np.array([0.,3.]),np.array([.05,.05]),
                      np.zeros((2,4)),.01,1,0.,0.,{}, {})
    fake=SimpleNamespace(solve_context=lambda c,g:plan,
                         plant=SimpleNamespace(audit=lambda *args:({'valid':False},{})))
    with pytest.raises(Refused) as caught:
        solve_audited(fake,{'CL_des':.025},np.array([.05,.05]),tmp_path,'pre')
    assert caught.value.phase=='pre'
    assert len(caught.value.attempts)==2
    assert 'AUDIT_REJECTED' in str(caught.value)
    assert (tmp_path/'pre_0.npz').is_file() and (tmp_path/'pre_1.npz').is_file()


def test_tail_solve_enforces_current_tracking_priority(system):
    ocp,_,_=system
    c=nominal(ocp)
    c['t']=1197.
    c['CL_des']=float(ocp.plant.base.quantities(c['x'],c['alpha_prev'],c['source'],c['target'],1.)[4])
    plan=ocp.solve_context(c)
    assert plan.success,plan.summary()
    assert plan.alpha_nodes[0]==c['alpha_prev'] and plan.times[-1]==3.
    assert plan.terms['current_tracking_cost']<=plan.stages['allowed_tracking_cost']+ocp.cfg['acceptance_violation']
    audit,_=ocp.plant.audit(c,plan.alpha_nodes,plan.times,c['CL_des'])
    assert audit['valid'],audit
