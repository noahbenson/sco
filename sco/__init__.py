# __init__.py

'''
The SCO Python library is a modular toolkit for predicting the response of the cortical surface to
visual stimuli.
'''

def model_data():
    '''
    sco.model_data() yields a mapping whose keys are model names and whose values are implementation
      plans for the given model name.
    '''
    if not hasattr(model_data, '_plans') or type(model_data._plans).__name__ != 'PMap':
        # Import relevant functions...
        import sco.impl.benson17, sco.impl.kay13, pyrsistent as pyr
        model_data._plans = pyr.m(benson17 = sco.impl.benson17.sco_plan,
                                  kay13    = sco.impl.kay13.sco_plan)
    return model_data._plans

def model_names():
    '''
    sco.model_names() yields a tuple of the names of the SCO models that can be built using the
       sco.build_model(...) function.
    '''
    return model_data().keys()

def build_model(model_name, force_exports=False):
    '''
    sco.build_model(name) builds an SCO model according to the given name and returns it. Valid
      model names correspond to valid SCO plans; see the sco.sco_plans dict.

    The following options may be given:
      * force_exports (default: True) specifies whether exporting functions that included in the
        model should be automatically run when the model is built. If this is set to True then
        all standard exports will complete before the model is returned.
    '''
    import sco.util, pimms
    if isinstance(model_name, basestring):
        model_name = model_name.lower()
        _plans = model_data()
        if model_name not in _plans: raise ValueError('Unknown mode: %s' % model_name)
        mdl = _plans[model_name]
        dat = mdl.nodes
        if force_exports: dat = dat.set('require_exports', sco.util.require_exports)
        return pimms.plan(dat)
    elif pimms.is_plan(model_name):
        if force_exports:
            dat = model_name.nodes
            dat = dat.set('require_exports', sco.util.require_exports)
            return pimms.plan(dat)
        else:
            return model_name
    elif pimms.is_map(model_name):
        dat = model_name
        if force_exports: dat = dat.set('require_exports', sco.util.require_exports)
        return pimms.plan(dat)
    else:
        raise ValueError('Unrecognized object type given as first argument to to build_model')
        

# Version information...
_version_major = 0
_version_minor = 3
_version_micro = 6
__version__ = "%s.%s.%s" % (_version_major, _version_minor, _version_micro)

description = 'Predict the response of the cortex to visual stimuli'
    
# The volume (default) calculation chain
def reload_sco():
    '''
    reload_sco() reloads the sco and all of its submodules; it returns the new sco module.
    '''
    import sys, sco, \
           sco.util, sco.anatomy, sco.stimulus, sco.contrast, sco.pRF, sco.impl, \
           sco.impl.benson17, sco.impl.kay13
    reload(sys.modules['sco.util.plot'])
    reload(sys.modules['sco.util.io'])
    reload(sys.modules['sco.util'])
    reload(sys.modules['sco.anatomy.core'])
    reload(sys.modules['sco.stimulus.core'])
    reload(sys.modules['sco.contrast.core'])
    reload(sys.modules['sco.pRF.core'])
    reload(sys.modules['sco.analysis.core'])
    reload(sys.modules['sco.anatomy'])
    reload(sys.modules['sco.stimulus'])
    reload(sys.modules['sco.pRF'])
    reload(sys.modules['sco.contrast'])
    reload(sys.modules['sco.analysis'])
    reload(sys.modules['sco.impl.benson17'])
    reload(sys.modules['sco.impl.kay13'])
    reload(sys.modules['sco.impl'])    
    return reload(sys.modules['sco'])

