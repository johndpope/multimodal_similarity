import pickle as pkl

f = open('/mnt/work/Stanford40/ImageSplits/actions.txt', 'r')
count = 0
lines = f.read().strip().split('\n')
lines = [l.split('\t')[0] for l in lines[1:]]

print lines

label2num = {}
num2label = {}
count = 0
for l in lines:
    label2num[l] = count
    num2label[count] = l
    count += 1

# build dictionary
text_dict = {}
count = 0
for l in lines:
    for ll in l.split('_'):
        if ll not in text_dict:
            text_dict[ll] = count
            count += 1

label = {'label2num': label2num, 'num2label': num2label, 'text_dict':text_dict}
pkl.dump(label, open('/mnt/work/Stanford40/results/label.pkl', 'w'))
