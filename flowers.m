[data, type] = read_iris_data();
type = type - 1;
save('flowers_data.mat', 'data', 'type')
