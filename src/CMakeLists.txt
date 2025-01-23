include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(matrix matrix.cpp)
add_library(stats pearson.cpp spearman.cpp chi2.cpp anova.cpp kruskal.cpp ttest.cpp mwu.cpp)

