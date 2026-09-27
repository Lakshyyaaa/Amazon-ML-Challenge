"""Full feature rows for France test pairs (input for France self-training)."""
from common import CACHE
from predict_test import score_test_features

F = score_test_features(lambda F: F, countries=['France'])
F.to_parquet(f'{CACHE}/feat_test_france.parquet', index=False)
print('saved', len(F))
