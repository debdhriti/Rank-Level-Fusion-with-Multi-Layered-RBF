import csv

def extractIris(count_of_each_type):
    X_train, Y_train, X_test, Y_test = [], [], [], []

    cnt_a, cnt_b, cnt_c = count_of_each_type,count_of_each_type,count_of_each_type
    size_of_tr = cnt_a + cnt_b + cnt_c
    va, vb, vc = [], [], []

    with open('Iris.csv', 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        cnt = 0
        for row in csv_reader:
            idx = int(row['Id'])
            input_array = []
            output_array = []

            input_array.append(int(10*float(row['SepalLengthCm'])))
            input_array.append(int(10*float(row['SepalWidthCm'])))
            input_array.append(int(10*float(row['PetalLengthCm'])))
            input_array.append(int(10*float(row['PetalWidthCm'])))

            if(row['Species']=="Iris-setosa"):
                output_array = [1,0,0]
            if(row['Species']=="Iris-versicolor"):
                output_array = [0,1,0]
            if(row['Species']=="Iris-virginica"):
                output_array = [0,0,1]

            if(row['Species']=="Iris-setosa"):
                va.append([input_array, output_array])
            if(row['Species']=="Iris-versicolor"):
                vb.append([input_array, output_array])
            if(row['Species']=="Iris-virginica"):
                vc.append([input_array, output_array])


    # random.shuffle(va)
    # random.shuffle(vb)
    # random.shuffle(vc)

    for i in range(len(va)):
        if i < cnt_a:
            X_train.append(va[i][0]), Y_train.append(va[i][1])
        else:
            X_test.append(va[i][0]), Y_test.append(va[i][1])

    for i in range(len(vb)):
        if i < cnt_b:
            X_train.append(vb[i][0]), Y_train.append(vb[i][1])
        else:
            X_test.append(vb[i][0]), Y_test.append(vb[i][1])


    for i in range(len(vc)):
        if i < cnt_c:
            X_train.append(vc[i][0]), Y_train.append(vc[i][1])
        else:
            X_test.append(vc[i][0]), Y_test.append(vc[i][1])
    
    data = va + vb + vc
    
    return X_train, Y_train, X_test, Y_test, size_of_tr, data