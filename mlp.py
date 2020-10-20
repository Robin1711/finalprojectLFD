
def mlp(data):
    X, Y = data
    print("Sommer's happy mlp fun time")
    print("With data:")
    print("X -", len(X), "-", type(X[1]), "-", X[1])
    print("Y -", len(Y), "-", type(Y[1]), "-", Y[1])


if __name__ == '__main__':
    data = {}
    mlp(data)