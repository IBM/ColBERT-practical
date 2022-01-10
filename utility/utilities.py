import glob
import errno
import os
import time
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def rel_link_last_file(path, link_basename, file_pattern):
    name = os.path.join(path, link_basename)
    list_of_files = glob.glob(f'{path}/{file_pattern}')
    latest_file = max(list_of_files, key=os.path.getctime)
    rel_path = os.path.relpath(latest_file, path)
    logger.info(f"Make a sym link of {rel_path} to {name}")
    try:
        os.symlink(rel_path, name)
    except OSError as e:
        if e.errno == errno.EEXIST:
            os.remove(name)
            os.symlink(rel_path, name)
        else:
            raise


def get_link_and_timestamp(path):
    while not os.path.exists(path):
        logger.info(f"waiting for {path}")
        time.sleep(0.5)
    try:
        timestamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%x %X')
        link = os.readlink(path)
    except FileNotFoundError:
        time.sleep(0.01)
        timestamp = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%x %X')
        link = os.readlink(path)
    return link, timestamp


def get_file_new_link_and_timestamp(path, prev_time="--"):
    # will wait until getting a new timestamp for the file
    while not os.path.exists(path):
        logger.info(f"waiting for {path}")
        time.sleep(0.5)
    try:
        new_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%x %X')
        link = os.readlink(path)
    except FileNotFoundError:
        new_time = prev_time
    while new_time == prev_time:
        logger.info("waiting for a newer model")
        time.sleep(0.5)
        try:
            new_time = datetime.fromtimestamp(os.path.getmtime(path)).strftime('%x %X')
            link = os.readlink(path)
        except FileNotFoundError:
            new_time = prev_time
    return link, new_time


