import numpy as np

# OG
def overlook(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        if k%100 == 0 : print(k)
        for i in range(length):
            for j in range(length):
                if(data[k][i]> data[k][j]):
                    adjmatrix[k][i][j] = 1
                else:
                    adjmatrix[k][i][j] = 0
    return adjmatrix


# WOG
def overlookg(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        for i in range(length):
            for j in range(i+1,length):
                if(data[k][i] > data[k][j]):
                    adjmatrix[k][i][j] = (data[k][i] - data[k][j])/abs(j-i)
                    adjmatrix[k][j][i] = -1*(data[k][i] - data[k][j]) / abs(j-i)
                elif data[k][i] == data[k][j]:
                    adjmatrix[k][i][j] = adjmatrix[k][j][i] = 0
                else:
                    adjmatrix[k][i][j] = -1*(data[k][j]-data[k][i])/abs(j-i)
                    adjmatrix[k][j][i] = (data[k][j]- data[k][i])/abs(j-i)

    return adjmatrix

# H
def LPhorizontal_h(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for i in range(samples):
        for k in range(length):
            Y = np.zeros((1, 1))
            for l in range(k+1,length):
                if abs(k-l) <= 1:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    if Y[0][0]<data[i][0]:
                        Y[0][0] = data[i][0]
                        Y = sorted(Y)
                elif data[i][k]>Y[0][0] and data[i][l]>Y[0][0]:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = data[i][l]
                    Y = sorted(Y)
    return adjmatrix


# LH
def LPhorizontal_lh(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for i in range(samples):
        for k in range(length):
            Y = np.zeros((1, 3))
            for l in range(k+1,length):
                if abs(k-l) <= 3:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    if Y[0][0]<data[i][0]:
                        Y[0][0] = data[i][0]
                        Y = sorted(Y)
                elif data[i][k]>Y[0][0] and data[i][l]>Y[0][0]:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = data[i][l]
                    Y = sorted(Y)
    return adjmatrix


# V
def LPvisibility_v(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    xlmatrix = np.zeros((samples, length,length))
    for k in range(samples):
        for i in range(length):
            for j in range(i, length):
                xlmatrix[k][i][j] = (data[k][j]-data[k][i])/abs(i-j+0.00001)
    for i in range(samples):
        for k in range(length):
            Y = np.ones((1, 1))
            Y = -1000*Y
            for l in range(k+1,length):
                if abs(k-l) <= 1:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k]=1
                    if Y[0][0]<xlmatrix[i][k][l]:
                        Y[0][0] = xlmatrix[i][k][l]
                        Y = sorted(Y)
                elif Y[0][0]<xlmatrix[i][k][l]:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = xlmatrix[i][k][l]
                    Y = sorted(Y)
                else:
                    adjmatrix[i][k][l]=0
                    adjmatrix[i][l][k]=0
    return adjmatrix


# 0-V 2-LV
def LPvisibility_lv(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    xlmatrix = np.zeros((samples, length,length))
    for k in range(samples):
        for i in range(length):
            for j in range(i,length):
                xlmatrix[k][i][j] = (data[k][j]-data[k][i])/abs(i-j+0.00001)
    for i in range(samples):
        for k in range(length):
            Y = np.ones((1, 3))
            Y = -1000*Y
            for l in range(k+1,length):
                if abs(k-l) <= 3:
                    adjmatrix[i][k][l]=1
                    adjmatrix[i][l][k]=1
                    if Y[0][0]<xlmatrix[i][k][l]:
                        Y[0][0] = xlmatrix[i][k][l]
                        Y = sorted(Y)
                elif Y[0][0]<xlmatrix[i][k][l]:
                    adjmatrix[i][k][l] = 1
                    adjmatrix[i][l][k] = 1
                    Y[0][0] = xlmatrix[i][k][l]
                    Y = sorted(Y)
                else:
                    adjmatrix[i][k][l]=0
                    adjmatrix[i][l][k]=0
    return adjmatrix


# WNG
def overlook_WNG(data):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        for i in range(length - 1):
            if data[k][i+1] > data[k][i]:
                adjmatrix[k][i][i+1] = data[k][i] - data[k][i+1]
                adjmatrix[k][i+1][i] = -1*(data[k][i] - data[k][i+1])
            elif data[k][i+1] < data[k][i]:
                adjmatrix[k][i][i+1] = -1*(data[k][i+1] - data[k][i])
                adjmatrix[k][i+1][i] = data[k][i+1] - data[k][i]
            else:
                adjmatrix[k][i][i+1] = 0
                adjmatrix[k][i+1][i] = 0
    return adjmatrix

# WRG
def overlook_WRG(data, neigbhor = 2, probability = 10):
    samples = data.shape[0]
    length = data.shape[1]
    adjmatrix = np.zeros((samples, length, length))
    for k in range(samples):
        np.random.seed(1)
        random_number = np.random.randint(0, 100, (neigbhor, length))
        np.random.seed(1)
        rewired = np.random.randint(0, length, (neigbhor, length))
        for i in range(length-neigbhor+1):
            for j in range(neigbhor):
                if random_number[j][i] > probability:
                    if data[k][i+j] > data[k][i]:
                        adjmatrix[k][i][i+j] = (data[k][i] - data[k][i+j])
                        adjmatrix[k][i+j][i] = -1 * ((data[k][i] - data[k][i+j]))
                    elif data[k][i+j] < data[k][i]:
                        adjmatrix[k][i][i+j] = -1 * (data[k][i+j] - data[k][i])
                        adjmatrix[k][i+j][i] = (data[k][i+j] - data[k][i])
                    else:
                        adjmatrix[k][i][i+j] = 0
                        adjmatrix[k][i+j][i] = 0
                else:
                    if data[k][rewired[j][i]] > data[k][i]:
                        adjmatrix[k][i][rewired[j][i]] = (data[k][i] - data[k][rewired[j][i]])
                        adjmatrix[k][rewired[j][i]][i] = -1 * ((data[k][i] - data[k][rewired[j][i]]))
                    elif data[k][rewired[j][i]] < data[k][i]:
                        adjmatrix[k][i][rewired[j][i]] = -1 * (data[k][rewired[j][i]] - data[k][i])
                        adjmatrix[k][rewired[j][i]][i] = (data[k][rewired[j][i]] - data[k][i])
                    else:
                        adjmatrix[k][i][rewired[j][i]] = 0
                        adjmatrix[k][rewired[j][i]][i] = 0
        for i in range(length-neigbhor+1,length-1):
            for j in range(neigbhor):
                if random_number[j][i] > probability:
                    if data[k][length-neigbhor+j] > data[k][i]:
                        adjmatrix[k][i][length-neigbhor+1+j] = (data[k][i] - data[k][length-neigbhor+1+j])
                        adjmatrix[k][length-neigbhor+1+j][i] = -1 * ((data[k][i] - data[k][length-neigbhor+1+j]))
                    elif data[k][length-neigbhor+1+j] < data[k][i]:
                        adjmatrix[k][i][length-neigbhor+1+j] = -1 * (data[k][length-neigbhor+1+j] - data[k][i])
                        adjmatrix[k][length-neigbhor+1+j][i] = (data[k][length-neigbhor+1+j] - data[k][i])
                    else:
                        adjmatrix[k][i][length-neigbhor+1+j] = 0
                        adjmatrix[k][length-neigbhor+1+j][i] = 0
                else:
                    if data[k][rewired[j][i]] > data[k][i]:
                        adjmatrix[k][i][rewired[j][i]] = (data[k][i] - data[k][rewired[j][i]])
                        adjmatrix[k][rewired[j][i]][i] = -1 * ((data[k][i] - data[k][rewired[j][i]]))
                    elif data[k][rewired[j][i]] < data[k][i]:
                        adjmatrix[k][i][rewired[j][i]] = -1 * (data[k][rewired[j][i]] - data[k][i])
                        adjmatrix[k][rewired[j][i]][i] = (data[k][rewired[j][i]] - data[k][i])
                    else:
                        adjmatrix[k][i][rewired[j][i]] = 0
                        adjmatrix[k][rewired[j][i]][i] = 0
    return adjmatrix
