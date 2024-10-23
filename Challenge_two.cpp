#include <Eigen/Dense>
#include <iostream>
#include <string>
#include <fstream>
#include <iomanip>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include <random>



using namespace Eigen;
using namespace std;


// Funzione per caricare un'immagine
unsigned char* loadImage(const std::string& filename, int& width, int& height, int& channels, int desired_channels = STBI_grey) {
    unsigned char* img = stbi_load(filename.c_str(), &width, &height, &channels, desired_channels);
    if (img == nullptr) {
        cerr << "Errore nel caricamento dell'immagine: " << filename << endl;
        exit(EXIT_FAILURE);
    }
    return img;
}

// Funzione per convertire un'immagine in una matrice di Eigen
MatrixXd imageToEigenMatrix(const unsigned char* img, int rows, int cols) {
    MatrixXd imgMatrix(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            
            imgMatrix(i, j) = static_cast<double>(img[i * cols + j]);
            //aggiungo nel codice il clampig ai fini di evitare che i valori siano fuori dal range 0-255
            if(imgMatrix(i,j)>255.0) imgMatrix(i,j)=255.0;
            else if(imgMatrix(i,j)<0.0) imgMatrix(i,j)=0.0;
            

        }
    }
    return imgMatrix;
}
MatrixXd productForTranposed(const MatrixXd A)
{
    MatrixXd B ;
    return B= A * A.transpose();
}
// Funzione per salvare una matrice in formato Matrix Market (MTX)
void saveMatrixMarket(const MatrixXd& matrix, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Errore nell'apertura del file: " << filename << std::endl;
        return;
    }

    file << "%%MatrixMarket matrix coordinate real general\n";
    file << matrix.rows() << " " << matrix.cols() << " " << matrix.nonZeros() << "\n";

    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (matrix(i, j) != 0) {
                file << i + 1 << " " << j + 1 << " " << matrix(i, j) << "\n"; // 1-based indexing
            }
        }
    }

    file.close();
    std::cout << "Matrice salvata con successo in formato Matrix Market: " << filename << std::endl;
}
int countNonZeroEntries(const MatrixXd& matrix) {
    return (matrix.array() != 0).count();
}
void extractAndReport(const MatrixXd& U, const MatrixXd& V, const VectorXd& singularValues, int k) {
    MatrixXd C = U.leftCols(k);
    MatrixXd D = V.leftCols(k) * singularValues.head(k).asDiagonal();

    int nonZeroC = countNonZeroEntries(C);
    int nonZeroD = countNonZeroEntries(D);

    std::cout << "For k = " << k << ":\n";
    std::cout << "Number of nonzero entries in C: " << nonZeroC << "\n";
    std::cout << "Number of nonzero entries in D: " << nonZeroD << "\n";

    // Compute the compressed image
    MatrixXd compressedImage = C * D.transpose();

    // Normalize the compressed image to the range [0, 255]
    compressedImage = (compressedImage.array() - compressedImage.minCoeff()) / (compressedImage.maxCoeff() - compressedImage.minCoeff()) * 255.0;

    // Convert the compressed image to unsigned char
    unsigned char* compressedImg = new unsigned char[compressedImage.size()];
    for (int i = 0; i < compressedImage.rows(); ++i) {
        for (int j = 0; j < compressedImage.cols(); ++j) {
            compressedImg[i * compressedImage.cols() + j] = static_cast<unsigned char>(compressedImage(i, j));
        }
    }

    // Save the compressed image as a PNG file
    std::string filename = "compressed_image_k" + std::to_string(k) + ".png";
    stbi_write_png(filename.c_str(), compressedImage.cols(), compressedImage.rows(), 1, compressedImg, compressedImage.cols());

    delete[] compressedImg;
}
MatrixXd createCheckerboard(int size, int blockSize) {
    MatrixXd board(size, size);
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            // Calcola quale colore assegnare in base alla posizione
            int blockRow = i / blockSize;
            int blockCol = j / blockSize;
            board(i, j) = ((blockRow + blockCol) % 2)*255;
        }
    }
    return board;
}

int main() {
    // Caricamento dell'immagine
    int width, height, channels;
    unsigned char* img = loadImage("Einstein.jpg", width, height, channels);

    int rows = height;
    int cols = width;
    cout << "Size of the matrix: " << rows << " x " << cols << endl;

    // Conversione dell'immagine in una matrice di Eigen
    MatrixXd imgMatrix = imageToEigenMatrix(img, rows, cols);
    //Ora chiamo  la funzione productForTransposed che riceve in input la matrice imgMatrix
    //e restituisce la matrice prodotto tra imgMatrix e la sua trasposta
    MatrixXd imgMatrixForTrasposed = productForTranposed(imgMatrix);
    //Salvo la matrice in formato Matrix Market
    saveMatrixMarket(imgMatrixForTrasposed, "matrixChallenge.mtx");
    //ora verifico se la matrice trasposta è uguale alla matrice originale
    bool isSymmetricA2 = imgMatrixForTrasposed.isApprox(imgMatrixForTrasposed.transpose());
    std::cout << "Is A2 symmetric? " << (isSymmetricA2 ? "Yes" : "No") << std::endl;
    //e ora mando in stampa la norma della matrice 
    cout << "Norma della matrice: " << imgMatrixForTrasposed.norm() << endl;
    //Per il secondo task ci è stato richiesto di risolvere l'eigenvalue problem la 
    //la prima operazione di cui dobbiamo occuparci è quella di calcolare gli
    //autovalori della nostra matrice 
    // Libera la memoria allocata per l'immagine
    Eigen::SelfAdjointEigenSolver<MatrixXd> eigenSolver(imgMatrixForTrasposed);
    //Ovviamente ci si assiucra che il calcolo sia andato a buon fine
    if (eigenSolver.info() != Eigen::Success) {
    cerr << "Eigenvalue decomposition failed." << endl;
    exit(EXIT_FAILURE);
    }
    //Ora stampo a monitor i primi due valori singolari
    cout << "Il più grande valore singolare: " << sqrt(eigenSolver.eigenvalues().reverse()(0)) << endl;
    cout << "Il secondo più grande valore singolare: " << sqrt(eigenSolver.eigenvalues().reverse()(1)) << endl;
    
    // Esegui la decomposizione SVD
    JacobiSVD<MatrixXd> svd(imgMatrix, ComputeThinU | ComputeThinV);
    MatrixXd U = svd.matrixU();
    MatrixXd V = svd.matrixV();
    VectorXd singularValues = svd.singularValues();

    // Calcola la norma euclidea dei valori singolari
    double norm = singularValues.norm();
    std::cout << "Norma euclidea della matrice diagonale dei valori singolari: " << norm << std::endl;
    //e con questo sono arrivato al task numero 7
    // Estrai e riporta per k = 40 e k = 80
    extractAndReport(U,V,singularValues,10);
    extractAndReport(U,V,singularValues,20);
    extractAndReport(U, V, singularValues, 40);
    extractAndReport(U, V, singularValues, 80);

    //Creo la scacchiera per eseguire il task 8 (200 x 200)
    MatrixXd checkerboardMatrix = createCheckerboard(200, 25);
    //Ora stampo a video la norma euclidea della matrice checkerboardMatrix
    cout << "Norma euclidea della matrice scacchiera: " << checkerboardMatrix.norm() << endl;
    rows=checkerboardMatrix.rows();
    cols=checkerboardMatrix.cols();
    unsigned char* checkerboard = new unsigned char[rows*cols];
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            checkerboard[i * cols + j] = static_cast<unsigned char>(checkerboardMatrix(i, j));
        }
    }
    //Per il task 9 devo sottoporre la chessboard ad un disturbo [-50,+50 ]
    unsigned char* noisy_checkerboard = new unsigned char[rows * cols];
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(-50, 50);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            noisy_checkerboard[i * cols + j] = std::clamp(checkerboard[i * cols + j] + dis(gen), 0, 255);
        }
    }

    // Salvataggio dell'immagine rumorosa nel file noisy_image.png
    stbi_write_png("noisy_checkerboard.png", cols, rows, 1, noisy_checkerboard, cols);

    // Salvataggio della immagine smoothed
    stbi_write_png("checkerboard.png", cols, rows, 1,checkerboard, cols);
    //Per il task 11 devo applicare la SVD alla eigen matrix corrispondente alla matrice rumorosa:
    MatrixXd noisy_checkerboardMatrix = imageToEigenMatrix(noisy_checkerboard, rows, cols);
    JacobiSVD<MatrixXd> svd_noisy_checkerboardMatrix(noisy_checkerboardMatrix, ComputeThinU | ComputeThinV);
    MatrixXd U_noisy_checkerboardMatrix = svd_noisy_checkerboardMatrix.matrixU();
    MatrixXd V_noisy_checkerboardMatrix = svd_noisy_checkerboardMatrix.matrixV();
    VectorXd singularValues_noisy_checkerboardMatrix = svd_noisy_checkerboardMatrix.singularValues();
    // Estrai e riporta per k = 5 e k = 10
    extractAndReport(U_noisy_checkerboardMatrix, V_noisy_checkerboardMatrix, singularValues_noisy_checkerboardMatrix, 5);
    extractAndReport(U_noisy_checkerboardMatrix, V_noisy_checkerboardMatrix, singularValues_noisy_checkerboardMatrix, 10);
    
    


  
    

    stbi_image_free(img);

    

    return 0;
}