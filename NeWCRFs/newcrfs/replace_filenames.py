f = open("./syns_patches/splits/val_files.txt", "r")
newlines = []
for i in f:
    part1, part2 = i.split('images/')
    # print(part2, part2[-8])
    part2 = part2.replace('/', '/images/')
    combined = part1+part2
    print(combined)
    newlines.append(combined)


f = open("./syns_patches/splits/val_files1.txt", "w")
for lines in newlines:
    f.write(lines)
f.close()