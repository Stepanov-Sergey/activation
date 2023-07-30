'''

Функция L1 (Манхэттенское расстояние):

Пример использования:

point1 = [1, 2, 3]
point2 = [4, 5, 6]
l1_dist = l1_distance(point1, point2)
print("L1 distance:", l1_dist)
'''
def l1_distance(x, y):
    distance = 0
    for i in range(len(x)):
        distance += abs(x[i] - y[i])
 return distance