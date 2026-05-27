#ifndef MATRIX_HPP
#define MATRIX_HPP

#include <vector>
#include <stdexcept>
#include <random>

class Matrix {
  private:
    std::vector<std::vector<double>> data;
    size_t rows, cols;
  public:
    Matrix() {}
    Matrix(size_t rows, size_t cols) : rows(rows), cols(cols) {
      data.resize(rows, std::vector<double>(cols, 0.));
    }

    size_t getRows() const {
      return this->rows;
    }
    size_t getCols() const {
      return this->cols;
    }

    double& operator()(size_t i, size_t j) {
      if(i >= this->rows || j >= this->cols) {
        throw std::out_of_range("Index out of bounds");
      }

      return this->data[i][j];
    }

    const double& operator()(size_t i, size_t j) const {
      if(i >= this->rows || j >= this->cols) {
        throw std::out_of_range("Index out of bounds");
      }

      return this->data[i][j];
    }

    Matrix& fillRandom(double min = -1., double max = 1.) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<> dist(min, max);

      for(size_t i=0; i<this->rows; i++) {
        for(size_t j=0; j<this->cols; j++) {
          // this->data[i][j] = dist(gen);
          (*this)(i,j) = dist(gen);
        }
      }


      return *this;
    }

    Matrix operator*(double scalar) const {
      Matrix res(this->rows, this->cols);
      for(size_t i=0; i<this->rows; i++){
        for(size_t j=0; j<this->cols; j++) {
          res(i, j) = (*this)(i, j) * scalar;
        }
      }

      return res;
    }

    Matrix& operator*=(double scalar) {
      for(size_t i=0; i<this->rows; i++) {
        for(size_t j=0; j<this->cols; j++) {
          (*this)(i, j) *= scalar;
        }
      }
    }

};

#endif