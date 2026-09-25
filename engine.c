#include <stdio.h>

/* L = 2*w1^2 + w2^2 - 8*w1 + 6*w2 + 15 */
double grad_w1(double w1) { return 4.0 * w1 - 8.0; }
double grad_w2(double w2) { return 2.0 * w2 + 6.0; }

double loss(double w1, double w2) {
  return 2.0 * w1 * w1 + w2 * w2 - 8.0 * w1 + 6.0 * w2 + 15.0;
}

int main(void) {
  double w1 = -10.0, w2 = 10.0;
  double lr = 0.05;
  double mom = 0.8;
  double v1 = 0.0, v2 = 0.0;
  int epochs = 100;

  printf("epoch | w1       | w2       | loss\n");
  printf("------------------------------------\n");

  for (int i = 0; i <= epochs; i++) {
    v1 = mom * v1 - lr * grad_w1(w1);
    v2 = mom * v2 - lr * grad_w2(w2);
    w1 += v1;
    w2 += v2;

    if (i % 5 == 0 || i == epochs)
      printf("%5d | %8.4f | %8.4f | %8.4f\n", i, w1, w2, loss(w1, w2));
  }

  printf("\ndone. w1=%.4f  w2=%.4f\n", w1, w2);
  return 0;
}
