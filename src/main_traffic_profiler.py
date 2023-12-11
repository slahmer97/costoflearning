from omegaconf import DictConfig
import hydra
from src.meta_network.greedy_policy import OnlyLearningPlanePolicy
from src.meta_network.mcrl import DQN
from src.meta_network.netqueue import ExperienceQueue
from src.meta_network.policy_selector import  get_policy_selector
from src.meta_network.tasks import Tasks
from src.simulation import Simulation


@hydra.main(config_path="../config", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    task_sampler = Tasks(cfg["traffic"])

    dqn = DQN(**cfg["agent"])

    greedy_policy = OnlyLearningPlanePolicy()
    ps_config = {
        "g_eps": 0.2,
        "g_eps_decay": 0,
        "g_eps_min": 0.002,
        "use": True,
        "end": 80
    }

    env = Simulation()
    import csv
    filename = f"./data/profile-traffic-0.csv"
    fo = open(filename, "w")
    header = ["taskid", "incoming1", "incoming2"]
    writer = csv.writer(fo)
    writer.writerow(header)
    fo = open(filename, "w")
    for i in range(len(task_sampler.tasks)):
        current_task = task_sampler.get_task(i)
        policy_selector = get_policy_selector(**cfg["policy_selector"])
        env.move_environment(**current_task)
        learning_queue_config = {
            "sim": env,
            "init": 0,
            "queue_type": "fifo"
        }
        learning_queue = ExperienceQueue(**learning_queue_config)
        for step in range(20000):
            ret, rew, additional_learning_res = env.rollout(dnn_policy=dqn, greedy_policy=greedy_policy,
                                                            policy_selector=policy_selector, k_steps=1,
                                                            meta_data={"cp": learning_queue})
            for (si, a, (r, c1, c2), sj, _, info) in ret:
                a1, a2 = info["incoming_traffic"]
                line = [i, a1, a2]
                writer.writerow(line)
            if step % 1000 == 0:
                print(f"task={i}  step={step}")
    fo.close()
if __name__ == "__main__":
    main()
