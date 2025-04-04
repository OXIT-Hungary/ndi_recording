import multiprocessing
import os
import queue
import subprocess
import time
from datetime import datetime
from pathlib import Path

import psutil
from tqdm import tqdm

from src.camera.camera_system import CameraSystem
from src.config import Config, load_config
from src.utils.logger import setup_logger

p = psutil.Process(os.getpid())
p.nice(value=-12)


def schedule(start_time: datetime, end_time: datetime) -> None:

    sleep_time = int((start_time - datetime.now()).total_seconds())
    if sleep_time <= 0:
        return

    hours, remainder = divmod(sleep_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    logger.info(
        f"Waiting for {hours:02d}:{minutes:02d}:{seconds:02d} (hh:mm:ss) until {end_time.strftime('%Y.%m.%d %H:%M:%S')}."
    )

    with tqdm(total=sleep_time, bar_format="{l_bar}{bar} [Elapsed: {elapsed}, Remaining: {remaining}]") as progress:
        for _ in range(int(sleep_time)):
            time.sleep(1)
            progress.update(1)

    logger.info(f"Finished waiting.")


def count_thread(config, event_stop: multiprocessing.Event) -> None:
    delta_time = int((config.end_time - config.start_time).total_seconds())

    try:
        with tqdm(total=delta_time, bar_format="{l_bar}{bar} [Elapsed: {elapsed}, Remaining: {remaining}]") as progress:
            i = 0
            while i < delta_time and not event_stop.is_set():
                time.sleep(1)
                progress.update(1)
                i += 1
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Terminating processes...")
    finally:
        event_stop.set()


def main(config: Config):

    camera_system = CameraSystem(config=config.camera_system, out_path=config.out_path)
    time.sleep(1)
    camera_system.start()

    time.sleep(60)
    print('ASD')
    camera_system.stop()
    print('ASD2')

    # proc_second = multiprocessing.Process(target=secondary_proc, args=(config, queue_frame, event_stop))
    # proc_second.start()

    logger.info("Finished recording.")


if __name__ == "__main__":
    import argparse

    multiprocessing.set_start_method("spawn")

    parser = argparse.ArgumentParser(prog="NDI Stream Recorder", description="Schedule a script to run based on time.")
    parser.add_argument("--config", type=str, help="Config path.", required=False, default="./config.yaml")
    parser.add_argument("--start_time", type=str, required=False, default=None)
    parser.add_argument("--end_time", type=str, required=False, default=None)
    parser.add_argument("--duration", type=str, required=False, default=None)

    args = parser.parse_args()
    cfg = load_config(file_path=args.config)

    cfg.schedule.start_time = (
        datetime.strptime(args.start_time, "%Y.%m.%d_%H:%M") if args.start_time else cfg.schedule.start_time
    )
    cfg.schedule.end_time = (
        datetime.strptime(args.end_time, "%Y.%m.%d_%H:%M") if args.end_time else cfg.schedule.end_time
    )
    cfg.schedule.duration = (
        datetime.strptime(args.duration, "%Y.%m.%d_%H:%M") if args.duration else cfg.schedule.duration
    )

    logger = setup_logger(log_dir=cfg.out_path)

    schedule(cfg.schedule.start_time, cfg.schedule.end_time)

    main(config=cfg)
