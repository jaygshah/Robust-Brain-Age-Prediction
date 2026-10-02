import torch
import torch.nn.functional as F


def order_loss(features, y, max_label_diff=None):
	"""ORDER loss (Eq. 4-5): Manhattan distance between L2-normalized features, weighted by |age_i - age_j|.

	max_label_diff: divide age differences by this constant (e.g. max_age - min_age) so weights are
	proportional to the age gap. If None, use the batch-wise scaling (w - min) / max that produced the
	WACV 2024 results.
	"""
	return _distance_regularizer(features, y, p=1.0, max_label_diff=max_label_diff)


def euclidean_loss(features, y, max_label_diff=None):
	# ORDER loss with the L2 distance (k = 2 in Tab. 5)
	return _distance_regularizer(features, y, p=2.0, max_label_diff=max_label_diff)


def order_loss_p(features, y, p=0.5, max_label_diff=None):
	# ORDER loss with a general Lk distance (fractional k in Tab. 5)
	return _distance_regularizer(features, y, p=p, max_label_diff=max_label_diff)


def _distance_regularizer(features, y, p, max_label_diff):
	if features.shape[0] < 2:
		return features.sum() * 0.0

	features = F.normalize(features, dim=1)
	y = y.float().view(features.shape[0], -1)

	# Pairwise feature distances and absolute label differences, upper triangle only
	distance = triu_up(torch.cdist(features, features, p=p))
	weights = triu_up(torch.cdist(y, y, p=1.0))
	weights = _scale_weights(weights, max_label_diff)

	return -torch.mean(distance * weights)


def _scale_weights(weights, max_label_diff):
	if max_label_diff is not None:
		return weights / max(float(max_label_diff), 1.0)

	weights_min, weights_max = torch.min(weights), torch.max(weights)
	if weights_max == 0:
		return weights
	return (weights - weights_min) / weights_max


def triu_up(dist):
	# Extract the upper triangular part of the distance matrix, excluding the diagonal
	a, b = dist.shape
	assert a == b
	indexes = torch.triu(torch.ones(a, b, dtype=torch.bool, device=dist.device), diagonal=1)
	return dist[indexes]


if __name__ == "__main__":
	x = torch.tensor([[1, 1], [2, 2], [3, 3]]).type(torch.FloatTensor)
	y = torch.tensor([[20], [40], [80]])

	print(order_loss(x, y, max_label_diff=85))
	print(order_loss(x, y))
