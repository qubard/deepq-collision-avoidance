with open('bestmodel') as f:
    content = f.readlines()
    for line in content:
        split = line.split(' ')
        mean = float(split[5])
        dev = float(split[7])
        low = mean - dev
        up = mean + dev
        print("("+str(int(split[1])) + "," + str(mean) + ")")
