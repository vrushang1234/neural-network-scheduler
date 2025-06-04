#include <iostream>
#include <fstream>
#include <iomanip>   // For std::setprecision
#include <cstdlib>   // For rand() and srand()
#include <ctime>     // For time()

int main() {
    int numEntries = 10000000;  // Change this to control number of entries

    std::ofstream outFile("./merge.csv");
    if (!outFile.is_open()) {
        std::cerr << "Failed to open file.\n";
        return 1;
    }

    std::srand(static_cast<unsigned>(std::time(nullptr)));  // Seed RNG

    // Write CSV headers
    outFile << "STUDENT ID,Grade\n";

    for (int i = 0; i < numEntries; ++i) {
        // Generate a 10-digit student ID (random between 1000000000 and 9999999999)
        long long studentID = 1000000000LL + std::rand() % 9000000000LL;

        // Generate grade with 15 decimal places
        double grade = static_cast<double>(std::rand()) / RAND_MAX * 100.0;

        outFile << studentID << ",";
        outFile << std::fixed << std::setprecision(15) << grade << "\n";
    }

    outFile.close();
    std::cout << "CSV file 'students.csv' created with " << numEntries << " entries.\n";

    return 0;
}

