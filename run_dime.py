import os
import jax
import time
import hydra
import wandb
import omegaconf
import traceback

# Set CUDA device from config - will be updated in main function

from common.buffers import DMCCompatibleDictReplayBuffer
from diffusion.dime import DIME
from omegaconf import DictConfig
from models.utils import is_slurm_job
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CallbackList
from models.actor_critic_evaluation_callback import EvalCallback
from models.episodic_return_callback import EpisodicReturnCallback
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
# set egl device = osmesa
os.environ['MUJOCO_GL'] = 'osmesa'



def _create_alg(cfg: DictConfig):
    import gymnasium as gym
    # try:
    #     import myosuite
    # except ImportError:
    #     print("myosuite not installed")
    #     pass
    
    # Environment setup with support for robomimic and ogbench
    assert not (cfg.if_robomimic and cfg.if_ogbench), "only one of robomimic or ogbench can be used"
    
    # standard gym environment
    training_env = gym.make(cfg.env_name)
    eval_env = make_vec_env(cfg.env_name, n_envs=1, seed=cfg.seed)
    env_name_split = cfg.env_name.split('/')
    rb_class = None
    if env_name_split[0] == 'dm_control':
        rb_class = DMCCompatibleDictReplayBuffer if env_name_split[1].split('-')[0] in ['humanoid', 'fish', 'walker', 'quadruped','finger'] else None

    tensorboard_log_dir = f"./logs/{cfg.wandb['group']}/{cfg.wandb['name']}/seed= + {str(cfg.seed)}/"
    eval_log_dir = f"./eval_logs/{cfg.wandb['group']}/{cfg.wandb['name']}/seed= + {str(cfg.seed)}/eval/"


    model = DIME(
        "MultiInputPolicy" if isinstance(training_env.observation_space, gym.spaces.Dict) else "MlpPolicy",
        env=training_env,
        model_save_path=None,
        save_every_n_steps=int(cfg.tot_time_steps / 100000),
        cfg=cfg,
        tensorboard_log=tensorboard_log_dir,
        replay_buffer_class=rb_class
    )

    # Create log dir where evaluation results will be saved
    os.makedirs(eval_log_dir, exist_ok=True)
    # Create callback that evaluates agent

    eval_callback = EvalCallback(
        eval_env,
        jax_random_key_for_seeds=cfg.seed,
        best_model_save_path=None,
        log_path=eval_log_dir,
        eval_freq=max(1000000 // cfg.log_freq, 1),
        n_eval_episodes=5, deterministic=True, render=False
    )
    
    # 创建情节回报记录回调
    episodic_return_callback = EpisodicReturnCallback(
        log_freq=5000, 
        verbose=1  # 显示详细信息
    )
    
    if cfg.wandb["activate"]:
        callback_list = CallbackList([eval_callback, episodic_return_callback, WandbCallback(verbose=0, gradient_save_freq=1000)])
    else:
        callback_list = CallbackList([eval_callback, episodic_return_callback])
    return model, callback_list


def initialize_and_run(cfg):
    cfg = hydra.utils.instantiate(cfg)
    seed = cfg.seed
    if cfg.wandb["activate"]:
        name = cfg.wandb["name"] + "_seed_" + str(seed)
        wandb_config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        wandb.init(
            settings=wandb.Settings(_service_wait=300),
            project=cfg.wandb["project"]+"-"+cfg.env_name,
            group=cfg.wandb["group"],
            name=name,
            config=wandb_config,
            entity=cfg.wandb["entity"],
            sync_tensorboard=False,
        )
        if is_slurm_job():
            print(f"SLURM_JOB_ID: {os.environ.get('SLURM_JOB_ID')}")
            wandb.summary['SLURM_JOB_ID'] = os.environ.get('SLURM_JOB_ID')
    model, callback_list = _create_alg(cfg)
    model.learn(total_timesteps=cfg.tot_time_steps, progress_bar=True, callback=callback_list)


@hydra.main(version_base=None, config_path="configs", config_name="base")
def main(cfg: DictConfig) -> None:
    # Set CUDA device from config
    os.environ['CUDA_VISIBLE_DEVICES'] = str(cfg.device)
    try:
        starting_time = time.time()
        if cfg.use_jit:
            initialize_and_run(cfg)
        else:
            with jax.disable_jit():
                initialize_and_run(cfg)
        end_time = time.time()
        print(f"Training took: {(end_time - starting_time)/3600} hours")
        if cfg.wandb["activate"]:
            wandb.finish()
    except Exception as ex:
        print("-- exception occured. traceback :")
        traceback.print_tb(ex.__traceback__)
        print(ex, flush=True)
        print("--------------------------------\n")
        traceback.print_exception(ex)
        if cfg.wandb["activate"]:
            wandb.finish()


if __name__ == "__main__":
    main()
