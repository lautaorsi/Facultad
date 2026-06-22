import random



def model():

    mu = random.gauss(0,1)

    y = random.gauss(mu,1)

    return (mu,y)

pairs = [model() for _ in range(100_000)]
print(sum(y > 2 for _, y in pairs) / len(pairs))
