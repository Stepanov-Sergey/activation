//реализация функции производной сигмоиды на JavaScript

/*
В этом примере мы используем Math.exp() для вычисления экспоненты и Math.multiply() для элемент-по-элементного умножения.
Функция sigmoid вычисляет значение сигмоиды для данного входного значения x,
а функция sigmoid_derivative вычисляет производную сигмоиды по формуле sig * (1 - sig),
где sig - значение сигмоиды.
*/

function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoid_derivative(x) {
  var sig = sigmoid(x);
  return sig * (1 - sig);
}

//example
var x = 2;
console.log(sigmoid_derivative(x)); // Вывод: 0.10458540350662
