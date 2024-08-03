import c3d

reader = c3d.Reader(open('../dataset/07_01.c3d', 'rb'))
for i, points, analog in reader.read_frames():
    print(points.shape, analog.shape)