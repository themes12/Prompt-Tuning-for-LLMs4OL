from configuration import BaseConfig
from src import EntityDatasetBuilderFactory
from datahandler import DataReader, DataWriter
from argparse import ArgumentParser

if __name__=="__main__":
    dataset_builder = EntityDatasetBuilderFactory(loader=DataReader)
    parser = ArgumentParser(description='Build datasets.')
    parser.add_argument('-kb_name', '--kb_name', type=str, choices=["wn18rr", "geonames", "umls"], help="Choose datasets name.", required=True)
    
    args = parser.parse_args()
    kb_name = args.kb_name
    if kb_name is not None:
        if kb_name == "wn18rr":
            config = BaseConfig(version=3).get_args(kb_name="wn18rr")
            wn_builder = dataset_builder(config=config)
            dataset_json, dataset_stats = wn_builder.build()
            DataWriter.write_json(data=dataset_json,
                                path=config.entity_path)
            DataWriter.write_json(data=dataset_stats,
                                path=config.dataset_stats)
        elif kb_name == "geonames":
            config = BaseConfig(version=3).get_args(kb_name="geonames")
            geo_builder = dataset_builder(config=config)
            dataset_json, dataset_stats = geo_builder.build()
            DataWriter.write_json(data=dataset_json,
                                path=config.entity_path)
            DataWriter.write_json(data=dataset_stats,
                                path=config.dataset_stats)
        elif kb_name == "umls":
            config = BaseConfig(version=3).get_args(kb_name="umls")
            umls_builder = dataset_builder(config=config)
            dataset_json, dataset_stats = umls_builder.build()
            DataWriter.write_json(data=dataset_json,
                                path=config.entity_path)
            DataWriter.write_json(data=dataset_stats,
                                path=config.dataset_stats)
    # for kb in list(dataset_json.keys()):
    #     DataWriter.write_json(data=dataset_json[kb],
    #                           path=BaseConfig(version=3).get_args(kb_name=kb.lower()).entity_path)
    #     DataWriter.write_json(data=dataset_stats,
    #                           path=BaseConfig(version=3).get_args(kb_name=kb.lower()).dataset_stats)
