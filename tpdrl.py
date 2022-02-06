from utils import ci_cycle, data_loader, utils
from stable_baselines3.common.monitor import Monitor

test_data_loader = data_loader.TestCaseExecutionDataLoader(
    "data/iofrol-additional-features.csv", "simple"
)
test_data = test_data_loader.load_data()
ci_cycle_logs = test_data_loader.pre_process()

experiment(
    mode=args.mode,
    algo=args.algo.upper(),
    test_case_data=ci_cycle_logs,
    episodes=int(args.episodes),
    start_cycle=conf.first_cycle,
    verbos=True,
    end_cycle=conf.first_cycle + conf.cycle_count - 1,
    model_path=conf.output_path,
    dataset_name="",
    conf=conf,
)
