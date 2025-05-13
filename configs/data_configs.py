from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'paris_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['paris_train'],
		'train_target_root': dataset_paths['paris_train'],
		'test_source_root': dataset_paths['paris_test'],
		'test_target_root': dataset_paths['paris_test'],
	},
	'places_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['places_train'],
		'train_target_root': dataset_paths['places_train'],
		'test_source_root': dataset_paths['places_test'],
		'test_target_root': dataset_paths['places_test'],
	},
	'celeba_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
}
