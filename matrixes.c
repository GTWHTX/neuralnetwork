#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

struct Matrix {
  int rows;
  int cols;
  double *data;
};

struct Layer {
  struct Matrix weights;
  struct Matrix biases;
  struct Matrix output;

  struct Matrix backprop_weights;
  struct Matrix backprop_biases;
  struct Matrix backprop_cache;

  struct Matrix dZ;
  struct Matrix dX;
};

struct NeuralNetwork {
  struct Layer input;
  struct Layer hidden_f;
  struct Layer hidden_s;
  struct Layer output;
};

const char *hiragana_labels[46] = {
    "aa", "chi", "ee", "fu",  "ha", "he", "hi", "ho", "ii", "ka",  "ke", "ki",
    "ko", "ku",  "ma", "me",  "mi", "mo", "mu", "na", "ne", "ni",  "nn", "no",
    "nu", "oo",  "ra", "re",  "ri", "ro", "ru", "sa", "se", "shi", "so", "su",
    "ta", "te",  "to", "tsu", "uu", "wa", "wo", "ya", "yo", "yu",
};

#define IO_MATRIX(m, f, op)                                                    \
  op((m).data, sizeof(double), (m).rows *(m).cols, (f))

struct Matrix create_matrix(int rows, int cols) {
  struct Matrix m;
  m.rows = rows;
  m.cols = cols;
  m.data = (double *)malloc(rows * cols * sizeof(double));
  return m;
}

void activate_relu(struct Matrix *m) {
  for (int i = 0; i < m->cols * m->rows; i++) {
    if (m->data[i] <= 0)
      m->data[i] = 0;
  }
}

void derivate_relu(struct Matrix *m) {
  for (int i = 0; i < m->cols * m->rows; i++) {
    if (m->data[i] <= 0)
      m->data[i] = 0;
    else
      m->data[i] = 1;
  }
}

void activate_sigmoid(struct Matrix *m) {
  for (int i = 0; i < m->cols * m->rows; i++) {
    m->data[i] = 1.0 / (1.0 + exp(-m->data[i]));
  }
}

void activate_softmax(struct Matrix *m) {
  int size = m->rows * m->cols;
  double max_val = 0.0;
  vDSP_maxvD(m->data, 1, &max_val, size);

  double neg_max = -max_val;
  vDSP_vsaddD(m->data, 1, &neg_max, m->data, 1, size);

  int n = size;
  vvexp(m->data, m->data, &n);

  double sum_val = 0.0;
  vDSP_sveD(m->data, 1, &sum_val, size);

  double inv_sum = 1.0 / sum_val;
  vDSP_vsmulD(m->data, 1, &inv_sum, m->data, 1, size);
}

void matrix_print(struct Matrix *m) {
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      printf("%6.2f\t", m->data[i * m->cols + j]);
    }
    printf("\n");
  }
  printf("\n");
}

void matrix_set(struct Matrix *m, int row, int col, double val) {
  int index = row * m->cols + col;
  m->data[index] = val;
}

void matrix_fill(struct Matrix *m, int input_size) {
  double scale = sqrt(2.0 / input_size);
  for (int i = 0; i < m->rows; i++) {
    for (int j = 0; j < m->cols; j++) {
      double rand_val = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
      m->data[i * m->cols + j] = rand_val * scale;
    }
  }
}

double matrix_get(struct Matrix *m, int row, int col) {
  int index = row * m->cols + col;
  return m->data[index];
}

void layer_forward(struct Layer *layer, struct Matrix *input,
                   size_t use_activation) {
  layer->backprop_cache = *input;
  memcpy(layer->output.data, layer->biases.data,
         layer->weights.cols * sizeof(double));
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, input->rows,
              layer->weights.cols, input->cols, 1.0, input->data, input->cols,
              layer->weights.data, layer->weights.cols, 1.0, layer->output.data,
              layer->output.cols);
  if (use_activation)
    activate_relu(&layer->output);
}

void layer_backward(struct Layer *layer, struct Matrix *backprop_output,
                    size_t use_activation) {
  for (int i = 0; i < layer->dZ.rows * layer->dZ.cols; i++) {
    if (use_activation) {
      if (layer->output.data[i] > 0)
        layer->dZ.data[i] = backprop_output->data[i];
      else
        layer->dZ.data[i] = 0.0;
    } else
      layer->dZ.data[i] = backprop_output->data[i];
  }
  memcpy(layer->backprop_biases.data, layer->dZ.data,
         layer->backprop_biases.cols * sizeof(double));
  memset(layer->backprop_weights.data, 0,
         layer->backprop_weights.rows * layer->backprop_weights.cols *
             sizeof(double));

  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              layer->backprop_cache.cols, layer->dZ.cols,
              layer->backprop_cache.rows, 1.0, layer->backprop_cache.data,
              layer->backprop_cache.cols, layer->dZ.data, layer->dZ.cols, 1.0,
              layer->backprop_weights.data, layer->backprop_weights.cols);

  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, layer->dZ.rows,
              layer->weights.rows, layer->dZ.cols, 1.0, layer->dZ.data,
              layer->dZ.cols, layer->weights.data, layer->weights.cols, 0.0,
              layer->dX.data, layer->dX.cols);
}

void layer_update(struct Layer *layer, double learning_rate) {
  int weights_size = layer->weights.rows * layer->weights.cols;
  int biases_size = layer->biases.rows * layer->biases.cols;
  cblas_daxpy(weights_size, -learning_rate, layer->backprop_weights.data, 1,
              layer->weights.data, 1);
  cblas_daxpy(biases_size, -learning_rate, layer->backprop_biases.data, 1,
              layer->biases.data, 1);
}

void layer_init(struct Layer *layer, size_t input_size, size_t neurons) {
  layer->output = create_matrix(1, neurons);
  layer->weights = create_matrix(input_size, neurons);
  layer->biases = create_matrix(1, neurons);
  layer->backprop_cache.data = NULL;
  layer->backprop_weights = create_matrix(input_size, neurons);
  layer->backprop_biases = create_matrix(1, neurons);
  layer->dZ = create_matrix(1, neurons);
  layer->dX = create_matrix(1, input_size);
  matrix_fill(&layer->weights, input_size);
  memset(layer->biases.data, 0,
         layer->biases.rows * layer->biases.cols * sizeof(double));
}

void layer_free(struct Layer *layer) {
  if (layer->weights.data != NULL)
    free(layer->weights.data);
  if (layer->biases.data != NULL)
    free(layer->biases.data);
  if (layer->output.data != NULL)
    free(layer->output.data);
  if (layer->backprop_biases.data != NULL)
    free(layer->backprop_biases.data);
  if (layer->backprop_weights.data != NULL)
    free(layer->backprop_weights.data);
  if (layer->dZ.data != NULL)
    free(layer->dZ.data);
  if (layer->dX.data != NULL)
    free(layer->dX.data);
}

void network_free(struct NeuralNetwork *nn) {
  layer_free(&nn->input);
  layer_free(&nn->hidden_f);
  layer_free(&nn->hidden_s);
  layer_free(&nn->output);
}

void save_network(struct NeuralNetwork *nn, const char *filename) {
  FILE *file = fopen(filename, "wb");
  if (!file)
    return;
  IO_MATRIX(nn->input.weights, file, fwrite);
  IO_MATRIX(nn->input.biases, file, fwrite);
  IO_MATRIX(nn->hidden_f.weights, file, fwrite);
  IO_MATRIX(nn->hidden_f.biases, file, fwrite);
  IO_MATRIX(nn->hidden_s.weights, file, fwrite);
  IO_MATRIX(nn->hidden_s.biases, file, fwrite);
  IO_MATRIX(nn->output.weights, file, fwrite);
  IO_MATRIX(nn->output.biases, file, fwrite);
  fclose(file);
}

void load_network(struct NeuralNetwork *nn, const char *filename) {
  FILE *file = fopen(filename, "rb");
  if (!file)
    return;
  IO_MATRIX(nn->input.weights, file, fread);
  IO_MATRIX(nn->input.biases, file, fread);
  IO_MATRIX(nn->hidden_f.weights, file, fread);
  IO_MATRIX(nn->hidden_f.biases, file, fread);
  IO_MATRIX(nn->hidden_s.weights, file, fread);
  IO_MATRIX(nn->hidden_s.biases, file, fread);
  IO_MATRIX(nn->output.weights, file, fread);
  IO_MATRIX(nn->output.biases, file, fread);
  fclose(file);
}

int load_dataset(const char *filename, struct Matrix *images, int **labels) {
  FILE *file = fopen(filename, "rb");
  if (file == NULL) {
    printf("[-] Failed to open dataset %s\n", filename);
    return 0;
  }
  int num_samples, input_size;
  fread(&num_samples, sizeof(int), 1, file);
  fread(&input_size, sizeof(int), 1, file);
  printf("[+] Loading dataset... Samples: %d, Input: %d\n", num_samples,
         input_size);
  *images = create_matrix(num_samples, input_size);
  *labels = (int *)malloc(num_samples * sizeof(int));
  for (int i = 0; i < num_samples; i++) {
    fread(&(*labels)[i], sizeof(int), 1, file);
    double *row_ptr = &images->data[i * input_size];
    fread(row_ptr, sizeof(double), input_size, file);
  }
  fclose(file);
  return num_samples;
}

int predict(struct NeuralNetwork *nn, struct Matrix *image) {
  layer_forward(&nn->input, image, 1);
  layer_forward(&nn->hidden_f, &nn->input.output, 1);
  layer_forward(&nn->hidden_s, &nn->hidden_f.output, 1);
  layer_forward(&nn->output, &nn->hidden_s.output, 0);

  activate_softmax(&nn->output.output);

  int best_idx = 0;
  double max_val = nn->output.output.data[0];

  for (int i = 1; i < 46; i++) {
    if (nn->output.output.data[i] > max_val) {
      max_val = nn->output.output.data[i];
      best_idx = i;
    }
  }
  return best_idx;
}

int main() {
  srand(time(NULL));
  for (int i = 0; i < 100; i++)
    rand();

  struct Matrix images;
  int *labels;
  int num_samples = load_dataset("dataset.bin", &images, &labels);
  if (num_samples == 0) {
    printf("[-] Failed to load dataset. Exiting...\n");
    return 1;
  }

  struct NeuralNetwork nn;
  layer_init(&nn.input, 16384, 512);
  layer_init(&nn.hidden_f, 512, 128);
  layer_init(&nn.hidden_s, 128, 64);
  layer_init(&nn.output, 64, 46);

  load_network(&nn, "hiragana_model.bin");

  struct Matrix input_data = create_matrix(1, 16384);
  struct Matrix output_loss = create_matrix(1, 46);
  double learning_rate = 0.001;
  int choice;

  while (1) {
    printf("\n--- Main Menu ---\n");
    printf("1. Train Network\n");
    printf("2. Predict Random Image\n");
    printf("3. Predict Custom Image\n");
    printf("4. Exit\n");
    printf("Select an option: ");

    if (scanf("%d", &choice) != 1) {
      int c;
      while ((c = getchar()) != '\n' && c != EOF)
        ;
      printf("[-] Invalid input. Please enter a number.\n");
      continue;
    }

    if (choice == 1) {
      int epochs;
      printf("[?] Enter number of epochs to train: ");
      if (scanf("%d", &epochs) != 1)
        epochs = 1000;

      FILE *log_file = fopen("training_loss.txt", "w");
      double average_loss = 0.0;

      printf("[+] Training started for %d epochs...\n", epochs);
      for (int i = 0; i < epochs; i++) {
        int rnd_index = rand() % num_samples;
        int current_label = labels[rnd_index];
        memcpy(input_data.data, &images.data[rnd_index * 16384],
               16384 * sizeof(double));

        layer_forward(&nn.input, &input_data, 1);
        layer_forward(&nn.hidden_f, &nn.input.output, 1);
        layer_forward(&nn.hidden_s, &nn.hidden_f.output, 1);
        layer_forward(&nn.output, &nn.hidden_s.output, 0);

        activate_softmax(&nn.output.output);

        double prob = nn.output.output.data[current_label];
        if (prob < 1e-15)
          prob = 1e-15;
        average_loss += -log(prob);

        memcpy(output_loss.data, nn.output.output.data, 46 * sizeof(double));
        output_loss.data[current_label] -= 1.0;

        layer_backward(&nn.output, &output_loss, 0);
        layer_backward(&nn.hidden_s, &nn.output.dX, 1);
        layer_backward(&nn.hidden_f, &nn.hidden_s.dX, 1);
        layer_backward(&nn.input, &nn.hidden_f.dX, 1);

        layer_update(&nn.output, learning_rate);
        layer_update(&nn.hidden_s, learning_rate);
        layer_update(&nn.hidden_f, learning_rate);
        layer_update(&nn.input, learning_rate);

        if (i % 100 == 0 && i > 0) {
          average_loss /= 100.0;
          printf("Epoch %d\tAverage Loss: %f\n", i, average_loss);
          if (log_file)
            fprintf(log_file, "%d %f\n", i, average_loss);
          average_loss = 0.0;
        }
      }
      if (log_file)
        fclose(log_file);
      save_network(&nn, "hiragana_model.bin");
      printf("[+] Training complete. Model saved to hiragana_model.bin\n");

    } else if (choice == 2) {
      int rnd_index = rand() % num_samples;
      memcpy(input_data.data, &images.data[rnd_index * 16384],
             16384 * sizeof(double));

      int prediction = predict(&nn, &input_data);
      int actual = labels[rnd_index];

      printf("\n[~] Network predicts class: %s | Actual class: %s\n",
             hiragana_labels[prediction], hiragana_labels[actual]);
      if (prediction == actual) {
        printf("[+] SUCCESS! The network guessed correctly.\n");
      } else {
        printf("[-] FAIL. The network made a mistake.\n");
      }

    } else if (choice == 4) {
      printf("[+] Exiting program. Freeing memory...\n");
      break;
    } else if (choice == 3) {
      FILE *f = fopen("custom_image.bin", "rb");
      if (!f) {
        printf("[-] Cannot find custom_image.bin. Run python script first!\n");
        continue;
      }
      fread(input_data.data, sizeof(double), 16384, f);
      fclose(f);
      int prediction = predict(&nn, &input_data);
      printf("\n[~] The Network looks at your drawing and says: It's %s!\n",
             hiragana_labels[prediction]);
    } else {
      printf("[-] Invalid choice. Try again.\n");
    }
  }

  free(input_data.data);
  free(output_loss.data);
  free(labels);
  free(images.data);
  network_free(&nn);

  return 0;
}
