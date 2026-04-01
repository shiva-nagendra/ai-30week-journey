#week 20 day 6
#simple attention intuition

import numpy as np

#words 
words = ["I","Love","AI"]

#pretend embeddings(random numbers)
np.random.seed(0)
embeddings = np.random.rand(3,4)

#similarity = dot product
scores = np.dot(embeddings, embeddings.T)

#softmax
exp_scores = np.exp(scores)
attention = exp_scores/np.sum(exp_scores, axis=1,keepdims=True)

print("Attention matrix:\n")
print(attention)