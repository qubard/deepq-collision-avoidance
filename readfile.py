with open('mar27') as f:
    content = f.readlines()
    for line in content:
        split = line.split(' ')
        if split[0] == 'Episode:':
            print("("+str(int(split[1])) + "," + split[3].split("\n")[0] + ")")
