from pymongo import MongoClient, TEXT, InsertOne, DeleteMany, ReplaceOne, UpdateOne
import pymongo
from bson.objectid import ObjectId
from bson import json_util
import json

client = MongoClient("mongodb+srv://user_1:ravioli@cluster0-fiuhd.azure.mongodb.net/test?retryWrites=true&w=majority")
db = client.get_database('cache_trace')

"""
    gets a "graph key" object from minio
    - a dict of graph types, with graph's flag and
    parameters for each
    each graph should have:
        source (source of data - workload, algorithm, etc)
        graph_type (line, scatter, or histogram)
        minio_flag
        x_label
        y_label
        title
        algorithm_specific (bool, is graph type only 
            available for certain algorithms?)
"""
def get_graph_key():
    gr_types = db.graph_types
    key = gr_types.find_one({"name": "graph type key"})
    #json_key = json.dumps(key)
    print(key['-r'])
    return key

# -------------------------- copied from storage.py -----------------------------
"""
    get all existing configs from mongo;
    existing config doesn't necessarily mean there's
    already a trace that's been run with this config
"""
def get_all_configs():
    all_configs = db.configs
    configs = all_configs.find()
    for c in configs:
        yield c


def find_config(c_id):
    all_configs = db.configs
    json_cfg = json.dumps(
        all_configs.find_one({"_id": ObjectId(str(c_id))}), default=json_util.default, indent=4)
    #config = all_configs.find_one({"_id": c_id})
    return json_cfg

def insert_config(config, name):
    print("IN INSERT CONFIG")
    conf = db.configs
    conf_obj = conf.find_one({'name': name})
    ins =''
    all_traces = []
    if conf_obj == None:
        for traces in set(config['traces']):
            for trace_names in StatGen.generateTraceNames(traces):
               all_traces.append(os.path.basename(trace_names))  
               print(os.path.basename(trace_names))   
        # got an error saying a set cannot be encoded by json so I make it a set and then a list again. 
        # The reason it is cast as a set is to remove duplicates in a simple manner 
        all_traces = set(all_traces)
        all_traces =  list(all_traces)
        config['traces'] = all_traces
        config["name"] = name
    ins = str(conf.insert_one(config).inserted_id)
    return ins
# -------------------------- copied from storage.py -----------------------------



def catch_config_errors(config):
    valid_algs = ['alecar6', 'arc', 'arcalecar', 'cacheus', 
                    'dlirs', 'lecar', 'lfu', 'lirs', 'lirsalecar', 'lru', 'mru']
    invalid_algs = []
    for algs in set(config['algorithms']):
        if algs.lower() not in valid_algs:
            pretty.failure("invalid algorithm:"+ algs + " detected")
            invalid_algs.append(algs)
    invalid_cache_size = []
    for sizes in config['cache_sizes']:
        st = str(sizes)
        if sizes > 1.0:
            invalid_cache_size.append(sizes)
        if sizes <= 0:
            invalid_cache_size.append(sizes)

    return invalid_algs, invalid_cache_size


def catch_json_errors(conf):
    try:
        conf = json.loads(conf)
        return conf
    except json.JSONDecodeError as e:
        return False
