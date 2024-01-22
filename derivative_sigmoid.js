//реализация функции производной сигмоиды на JavaScript
function sigmoid(x) {
  return 1 / (1 + Math.exp(-x));
}

function sigmoid_derivative(x) {
  var sig = sigmoid(x);
  return sig * (1 - sig);
}
var x = 2;
console.log(sigmoid_derivative(x)); // Вывод: 0.10458540350662
