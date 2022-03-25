import argparse
import logging
import subprocess
import time
from os.path import join, exists
from os import makedirs, execv, stat
import sys
from tqdm import tqdm
from .wikidata_ids import WIKITILE_2_WIKIDATA_TSV_NAME, DEFAULT_NAME_SERVER_PORT, load_names, load_wikidata_ids

logger = logging.getLogger(__name__)
REDIS_PORT_FILE = "port.txt"
REDIS_NAME_SERVER_DIR = "redis_name_server"


def wait_for_redis(port):
    import redis
    while True:
        r = redis.StrictRedis(host="localhost", port=port)
        try:
            r.get("test")
            logger.info("Connected to redis")
            return r
        except (redis.exceptions.ConnectionError, redis.exceptions.ResponseError, redis.exceptions.TimeoutError) as e:
            logger.info(f"Error connecting to redis (error={e})")
            time.sleep(1.0)
            pass


class RedisWikidataNameService(object):
    """Simple api for grabbing wikidata titles from wikidata ids."""
    def __init__(self, redis):
        self._redis = redis

    def get(self, wikidata_id, default=None):
        res = self._redis.get(wikidata_id)
        return default if res is None else res.decode("utf-8")


def load_names_or_name_server(path, num_names_to_load, prefix):
    redis_port_file = join(path, REDIS_NAME_SERVER_DIR, prefix, REDIS_PORT_FILE)
    if exists(redis_port_file):
        import redis
        with open(redis_port_file, "rt") as fin:
            port = int(fin.read())
        try:
            r = redis.StrictRedis(host="localhost", port=port)
            r.get("test")
            return RedisWikidataNameService(redis=r), float('inf'), True
        except redis.exceptions.ConnectionError:
            pass
    # fallback to loading the file
    known_names = load_names(
        join(path, WIKITILE_2_WIKIDATA_TSV_NAME),
        num_names_to_load,
        prefix=prefix
    )
    return known_names, num_names_to_load, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("wikidata")
    parser.add_argument("--port", type=int, default=DEFAULT_NAME_SERVER_PORT)
    parser.add_argument("--prefix", type=str, default="enwiki")
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)
    redis_dir = join(args.wikidata, REDIS_NAME_SERVER_DIR, args.prefix)
    makedirs(redis_dir, exist_ok=True)
    conf_filename = join(redis_dir, "redis.conf")
    port_filename = join(redis_dir, REDIS_PORT_FILE)
    success_filename = join(redis_dir, "success")
    if exists(success_filename) and exists(join(redis_dir, "dump.rdb")) and stat(join(redis_dir, "dump.rdb")).st_size > 100:
        logger.info("Found pre-existing redis db, skipping name loading")
        loaded, ids = None, None
    else:
        logger.info("Loading names into memory, this can take a sec.")
        loaded = load_names(join(args.wikidata, WIKITILE_2_WIKIDATA_TSV_NAME), float('inf'), prefix=args.prefix)
        logger.info(f"Loaded {len(loaded)} names into memory, getting ids.")
        ids, name2index = load_wikidata_ids(args.wikidata, verbose=True)
        del name2index
    if not exists(conf_filename):
        with open(conf_filename, "wt") as fout:
            fout.write(f"""
    rdbcompression yes
    rdbchecksum yes
    protected-mode no
    dbfilename dump.rdb
    dir {redis_dir}
    logfile {join(redis_dir, 'redis-server.log')}
        """)
    cmd = ["redis-server", conf_filename, "--port", str(args.port)]
    with open(port_filename, "wt") as fout:
        fout.write(str(args.port) + "\n")
    if loaded is not None:
        p = subprocess.Popen(cmd)
        try:
            p.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
        assert p.poll() is None, "redis-server exited early."
        try:
            import redis
            r = wait_for_redis(args.port)
            logger.info(f"Uploading {len(loaded)} names to redis.")
            pipe = r.pipeline()
            for key, value in tqdm(loaded.items()):
                pipe.set(ids[key], value)
            logger.info(f"Executing redis command.")
            pipe.execute()
            logger.info(f"Saving redis to disk.")
            r.save()
            with open(success_filename, "wt") as fout:
                fout.write(f"Uploaded {len(loaded)}.\n")
            logger.info(f"Relaunching redis as standalone.")
            del loaded
            del r
        finally:
            p.kill()
    execv("/usr/bin/redis-server", cmd)


if __name__ == "__main__":
    main()
