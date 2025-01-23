#include <pybind11/pybind11.h>

#pragma once

namespace py = pybind11;

// Matrix class for capsulating numpy array.
class DataMatrix {
    public:
        DataMatrix(py::ssize_t rows, py::ssize_t cols) : m_rows(rows), m_cols(cols) {
            m_data = new double[(size_t) (rows * cols)];
            memset(m_data, 0, sizeof(double) * (size_t) (rows * cols));
        }

        DataMatrix(const DataMatrix &s) : m_rows(s.m_rows), m_cols(s.m_cols) {
            m_data = new double[(size_t) (m_rows * m_cols)];
            memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_rows * m_cols));
        }

        DataMatrix(DataMatrix &&s) noexcept : m_rows(s.m_rows), m_cols(s.m_cols), m_data(s.m_data) {
            s.m_rows = 0;
            s.m_cols = 0;
            s.m_data = nullptr;
        }

        ~DataMatrix() {
            delete[] m_data;
        }

        DataMatrix &operator=(const DataMatrix &s) {
            if (this == &s) {
                return *this;
            }
            delete[] m_data;
            m_rows = s.m_rows;
            m_cols = s.m_cols;
            m_data = new double[(size_t) (m_rows * m_cols)];
            memcpy(m_data, s.m_data, sizeof(double) * (size_t) (m_rows * m_cols));
            return *this;
        }

        DataMatrix &operator=(DataMatrix &&s) noexcept {
            if (&s != this) {
                delete[] m_data;
                m_rows = s.m_rows;
                m_cols = s.m_cols;
                m_data = s.m_data;
                s.m_rows = 0;
                s.m_cols = 0;
                s.m_data = nullptr;
            }
            return *this;
        }

        double operator()(py::ssize_t i, py::ssize_t j) const;

        double &operator()(py::ssize_t i, py::ssize_t j);

        double *data() { return m_data; }

        py::ssize_t rows() const { return m_rows; }
        py::ssize_t cols() const { return m_cols; }


    private:
        py::ssize_t m_rows;
        py::ssize_t m_cols;
        double *m_data;
};
