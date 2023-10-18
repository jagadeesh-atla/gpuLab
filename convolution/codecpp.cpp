#include <bits/stdc++.h>

using namespace std;

#define RADIUS 1
#define maxFilters 3

#define dimF  2 * RADIUS + 1

template<typename T>
std::ostream& operator<<(std::ostream& out, const vector<T>& vec) {
	if (vec.empty()) {out << "[]"; return out;}
	for (int i = 0;  i < vec.size(); ++i)
		out << "[ "[i != 0] << vec[i] << ",]"[i == vec.size() - 1];
	return out; 
}

/*
__global__ void convolution_2d(vector<int>& matrix, vector<int>& filter, vector<int>& result, int N) {
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	
	int start_r = row - RADIUS;
	int start_c = col - RADIUS;
	
	int temp = 0;
	for (int i = 0; i < RADIUS; ++i)
		for (int j = 0; j < RADIUS; ++j) 
			if ((start_r + i) >= 0 && (start_r + i) < N) 
				if ((start_c + j) >= 0 && (start_c + j) < N)
					temp += matrix[(start_r + i) * N + (start_c + j)] * filter[i* dimF + j];

	result[row * N + col] = temp;
} 
*/

void getFiltersFromFile(char* fileName, vector< vector<int> > filters) {
	ifstream file(fileName);
	if (! file.is_open()) {
		cerr << "Unable to open file " << fileName << ".\n ";
		return;
	}
	vector <int> currentFilter;
	int value;
	while (file >> value) {
		currentFilter.push_back(value);

		if (file.peek() == '\n') {
			filters.push_back(currentFilter);
			currentFilter.clear();
		}
	}	
	
	file.close();
	return;
}

int main(void) {
	vector< vector<int> > Xfilters, Hfilters, Ifilters;
	getFiltersFromFile("./filters/X.txt", Xfilters);
	getFiltersFromFile("./filters/H.txt", Hfilters);
	getFiltersFromFile("./filters/I.txt", Ifilters);

	cout << Xfilters << endl;

	return 0;
}

