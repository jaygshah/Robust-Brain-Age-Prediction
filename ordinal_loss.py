import torch
import torch.nn.functional as F
from torchmetrics.functional import pairwise_manhattan_distance

def order_loss(features, y):
	# Normalize features if there is more than one feature vector
	if features.shape[0] > 1:
		features = F.normalize(features, dim=1)
	
	# Compute pairwise Manhattan distances and extract upper triangular part
	distance = triu_up(pairwise_manhattan_distance(features, features))
	weights = triu_up(pairwise_manhattan_distance(y, y))
	
	# Check if there are any pairs with the same label
	has_same_label = 0 in weights
	if features.shape[0] > 1:
		# Normalize weights to be in the range [0, 1]
		weights_max, weights_min = torch.max(weights), torch.min(weights)
		if weights_min == weights_max == 0:
			weights_max = 1
		weights = ((weights - weights_min) / weights_max)

	# Compute weighted distance and loss
	distance = distance * weights
	loss = -torch.mean(distance)
	return loss

def euclidean_loss(features, y):
	# Normalize features if there is more than one feature vector
	if features.shape[0] > 1:
		features = F.normalize(features, dim=1)
	
	# Compute pairwise Euclidean distances and extract upper triangular part
	distance = triu_up(euclidean_distance(features, features))
	weights = triu_up(euclidean_distance(y, y))
	
	# Check if there are any pairs with the same label
	has_same_label = 0 in weights
	if features.shape[0] > 1:
		# Normalize weights to be in the range [0, 1]
		weights_max, weights_min = torch.max(weights), torch.min(weights)
		weights = ((weights - weights_min) / weights_max)
	
	# Compute weighted distance and loss
	distance = distance * weights
	loss = -torch.mean(distance)
	return loss

def order_loss_p(features, y, p=float(1/2)):
	# Normalize features if there is more than one feature vector
	if features.shape[0] > 1:
		features = F.normalize(features, dim=1)
	
	# Compute pairwise distances with parameter p and extract upper triangular part
	distance = triu_up(distance_p(features, features, p))
	weights = triu_up(distance_p(y, y, p))
	
	# Check if there are any pairs with the same label
	has_same_label = 0 in weights
	if features.shape[0] > 1:
		# Normalize weights to be in the range [0, 1]
		weights_max, weights_min = torch.max(weights), torch.min(weights)
		weights = ((weights - weights_min) / weights_max)
	
	# Compute weighted distance and loss
	distance = distance * weights
	loss = -torch.mean(distance)
	return loss

def distance_p(x, y, p):
	# Compute pairwise distances with parameter p
	x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
	distance = torch.cdist(x, y, p=p)
	return distance

def euclidean_distance(x, y):
	# Compute pairwise Euclidean distances
	x, y = x.type(torch.FloatTensor), y.type(torch.FloatTensor)
	distance = torch.cdist(x, y, p=2.0)
	return distance

def triu_up(dist):
	# Extract the upper triangular part of the distance matrix, excluding the diagonal
	a, b = dist.shape
	assert a == b
	indexes = torch.triu(torch.ones(a, b), diagonal=1).to(torch.bool)
	return dist[indexes]

if __name__ == "__main__":
	# Example usage of the loss functions
	x = torch.tensor([[1,  1], [2, 2], [3, 3]]).type(torch.FloatTensor)
	y = torch.tensor([[1], [1], [1]]).type(torch.FloatTensor)

	print(order_loss(x, y))
