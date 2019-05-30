#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <string>

using namespace cv;
using namespace std;

const double MAX_VALUE = 10e10;
const double MIN_VALUE = -10e10;

int Huang2(const vector<int>& data) {
	int first, last;
	for (first = 0; first < data.size() && data[first] == 0; first++)
		;
	for (last = data.size() - 1; last > first && data[last] == 0; last--)
		;
	if (first == last)
		return 0;

	vector<double>S(last + 1);
	vector<double>W(last + 1);
	S[0] = data[0];

	for (int i = max(1, first); i <= last; i++) {
		S[i] = S[i - 1] + data[i];
		W[i] = W[i - 1] + i * data[i];
	}

	double C = last - first;
	vector<double>Smu(last + 1 - first);
	for (int i = 1; i < Smu.size(); i++) {
		double mu = 1 / (1 + abs(i) / C);
		Smu[i] = -mu * log(mu) - (1 - mu) * log(1 - mu);
	}

	int bestThreshold = 0;
	double bestEntropy = MAX_VALUE;
	for (int threshold = first; threshold <= last; threshold++) {
		double entropy = 0;
		int mu = (int)round(W[threshold] / S[threshold]);
		for (int i = first; i <= threshold; i++)
			entropy += Smu[abs(i - mu)] * data[i];
		mu = (int)round((W[last] - W[threshold]) / (S[last] - S[threshold]));
		for (int i = threshold + 1; i <= last; i++)
			entropy += Smu[abs(i - mu)] * data[i];

		if (bestEntropy > entropy) {
			bestEntropy = entropy;
			bestThreshold = threshold;
		}
	}

	return bestThreshold;
}

int Yen(const vector<int>& data) {
	int threshold;
	int ih, it;
	double crit;
	double max_crit;
	vector<double> norm_histo(data.size());
	vector<double> P1(data.size());
	vector<double> P1_sq(data.size());
	vector<double> P2_sq(data.size());

	int total = 0;
	for (ih = 0; ih < data.size(); ih++)
		total += data[ih];

	for (ih = 0; ih < data.size(); ih++)
		norm_histo[ih] = (double)data[ih] / total;

	P1[0] = norm_histo[0];
	for (ih = 1; ih < data.size(); ih++)
		P1[ih] = P1[ih - 1] + norm_histo[ih];

	P1_sq[0] = norm_histo[0] * norm_histo[0];
	for (ih = 1; ih < data.size(); ih++)
		P1_sq[ih] = P1_sq[ih - 1] + norm_histo[ih] * norm_histo[ih];

	P2_sq[data.size() - 1] = 0.0;
	for (ih = data.size() - 2; ih >= 0; ih--)
		P2_sq[ih] = P2_sq[ih + 1] + norm_histo[ih + 1] * norm_histo[ih + 1];

	threshold = -1;
	max_crit = MIN_VALUE;
	for (it = 0; it < data.size(); it++) {
		crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? log(P1[it] * (1.0 - P1[it])) : 0.0);
		if (crit > max_crit) {
			max_crit = crit;
			threshold = it;
		}
	}
	return threshold;
}

int main(int argc, char **argv)
{
	string filename = "images/test3.png";

	Mat im = imread(filename, IMREAD_GRAYSCALE);
	
	int histSize = 256;
	float range[] = { 0, 256 };
	const float* histRange = { range };
	bool uniform = true;
	bool accumulate = false;

	Mat hist;
	calcHist(&im, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

	int threshold_huang2 = Huang2(hist);
	int threshold_yen = Yen(hist);

	Mat im_yen, im_huang2;
	threshold(im, im_huang2, threshold_huang2, 255, THRESH_BINARY);
	threshold(im, im_yen, threshold_yen, 255, THRESH_BINARY);

	int pos = filename.find(".png");
	string filename_to_huang2 = filename.substr(0, pos) + "_huang2.png";
	string filename_to_yen = filename.substr(0, pos) + "_yen.png";

	imwrite(filename_to_huang2, im_huang2);
	imwrite(filename_to_yen, im_yen);

	namedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", im);

	namedWindow("huagn2", WINDOW_AUTOSIZE);
	imshow("huagn2", im_huang2);

	namedWindow("yen", WINDOW_AUTOSIZE);
	imshow("yen", im_yen);
	
	waitKey(0);
	return 0;
}
