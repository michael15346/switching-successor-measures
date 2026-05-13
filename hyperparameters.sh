ICVF
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/icvf.py 
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/icvf.py --agent.alpha=3
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/icvf.py --agent.discount=0.995 --agent.alpha=3
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/icvf.py 


OneStep FB
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/onestep_fb.py --agent.alpha=0.03
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/onestep_fb.py --agent.alpha=0.03 
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/onestep_fb.py --agent.alpha=0.03 --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/onestep_fb.py --agent.alpha=0.1


HIQL
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/hiql.py 
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/hiql.py 
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/hiql.py --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/hiql.py


HIQL without high-level policy
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/hiql_nonhierarchical.py 
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/hiql_nonhierarchical.py 
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/hiql_nonhierarchical.py --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/hiql_nonhierarchical.py


HIQL with bilinear critic
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/hiql_bilinear.py 
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/hiql_bilinear.py 
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/hiql_bilinear.py --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/hiql_bilinear.py 


HIQL with intentions
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/hiql_intentions.py 
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/hiql_intentions.py 
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/hiql_intentions.py --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/hiql_intentions.py 


FB
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/fb.py 
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/fb.py 
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/fb.py --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/fb.py 


FB with high-level policy
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/fb_hierarchical.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_MODEL --agent.high_alpha=1e-3
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/fb_hierarchical.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_MODEL --agent.high_alpha=1e-3
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/fb_hierarchical.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_MODEL --agent.high_alpha=1e-3
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/fb_hierarchical.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_MODEL --agent.high_alpha=1e-3


FB pi-Switch without high-level policy
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/fbpiswitch_nonhierarchical.py --agent.orthonorm_coeff=1e-4
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/fbpiswitch_nonhierarchical.py --agent.orthonorm_coeff=1e-4
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/fbpiswitch_nonhierarchical.py --agent.orthonorm_coeff=1e-4 --agent.discount=0.995
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/fbpiswitch_nonhierarchical.py --agent.orthonorm_coeff=1e-4


FB pi-Switch
python main.py --env_name=ogbench-antmaze-medium-navigate-v0 --agent=agents/fbpiswitch.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_pi_SWITCH_REPRESENTATIONS
python main.py --env_name=ogbench-antmaze-large-navigate-v0 --agent=agents/fbpiswitch.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_pi_SWITCH_REPRESENTATIONS
python main.py --env_name=ogbench-antmaze-giant-navigate-v0 --agent=agents/fbpiswitch.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_pi_SWITCH_REPRESENTATIONS
python main.py --env_name=ogbench-antmaze-teleport-navigate-v0 --agent=agents/fbpiswitch.py --train_steps=500_000 --agent.frozen_path=PATH_TO_PRETRAINED_FB_pi_SWITCH_REPRESENTATIONS


