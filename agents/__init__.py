from agents.fb import FBAgent
from agents.icvf import ICVFAgent
from agents.onestep_fb import OneStepFBAgent
from agents.hiql import HIQLAgent
from agents.hiql_nonhierarchical import HIQLNonHierarchicalAgent
from agents.hiql_bilinear import HIQLBilinearAgent
from agents.hiql_intentions import HIQLwithIntentionsAgent
from agents.fbpiswitch_nonhierarchical import FBpiSwitchNonHierarchicalAgent
from agents.fbpiswitch import FBpiSwitchAgent
from agents.fb_hierarchical import FBHierarchicalAgent


agents = dict(
    fb=FBAgent,
    icvf=ICVFAgent,
    onestep_fb=OneStepFBAgent,
    hiql=HIQLAgent,
    iql=HIQLNonHierarchicalAgent,
    iqlbilinear=HIQLBilinearAgent,
    iqlintentions=HIQLwithIntentionsAgent,
    fbpiswitch_nonh=FBpiSwitchNonHierarchicalAgent,
    fbpiswitch=FBpiSwitchAgent,
    hfb=FBHierarchicalAgent,
)



