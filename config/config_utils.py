import yaml

def load_config(config_yml):
	with open(config_yml, "r") as f:
		data_configs = yaml.load(f, Loader=yaml.FullLoader)

	return data_configs
