/*#define _CRT_SECURE_NO_WARNINGS*/

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

vector<pair<double, double>> Huang2_F(const vector<int>& data) {
	vector<pair<double, double>>f;
	int first, last;
	for (first = 0; first < data.size() && data[first] == 0; first++)
		;
	for (last = data.size() - 1; last > first && data[last] == 0; last--)
		;
	if (first == last) {
		f.push_back(make_pair(0, 0));
		return f;
	}

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

	for (int threshold = first; threshold <= last; threshold++) {
		double entropy = 0;
		int mu = (int)round(W[threshold] / S[threshold]);
		for (int i = first; i <= threshold; i++)
			entropy += Smu[abs(i - mu)] * data[i];
		mu = (int)round((W[last] - W[threshold]) / (S[last] - S[threshold]));
		for (int i = threshold + 1; i <= last; i++)
			entropy += Smu[abs(i - mu)] * data[i];

		f.push_back(make_pair(threshold, entropy));
	}

	return f;
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

vector<double> Yen_F(const vector<int>& data) {
	vector<double>f;
	int ih, it;
	double crit;
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

	for (it = 0; it < data.size(); it++) {
		crit = -1.0 * ((P1_sq[it] * P2_sq[it]) > 0.0 ? log(P1_sq[it] * P2_sq[it]) : 0.0) + 2 * ((P1[it] * (1.0 - P1[it])) > 0.0 ? log(P1[it] * (1.0 - P1[it])) : 0.0);
		f.push_back(crit);
	}
	return f;
}

int main(int argc, char **argv)
{
	string filename = "images/test.png";

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

	vector<pair<double, double>>huang2_f = Huang2_F(hist);
	vector<double>huang2_f_data(huang2_f.size());
	for (int i = 0; i < huang2_f_data.size(); i++)
		huang2_f_data[i] = huang2_f[i].second;

	vector<double>yen_f = Yen_F(hist);

	// Draw the histogram
	int hist_w = 512;
	int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage1(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImage2(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage1.rows, NORM_MINMAX, -1, Mat());
	normalize(huang2_f_data, huang2_f_data, 0, histImage1.rows, NORM_MINMAX, -1, Mat());
	normalize(yen_f, yen_f, 0, histImage2.rows, NORM_MINMAX, -1, Mat());

	for (int i = 1; i < histSize; i++)
	{
		line(histImage1, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage2, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))), Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	for (int i = 1; i < huang2_f.size(); i++)
	{
		line(histImage1, Point(huang2_f[i - 1].first * 2, hist_h - huang2_f_data[i - 1]), Point(huang2_f[i].first * 2, hist_h - huang2_f_data[i]),
			Scalar(0, 0, 255), 2, 5, 0);
	}

	for (int i = 1; i < yen_f.size(); i++)
	{
		line(histImage2, Point(i * 2, hist_h - yen_f[i - 1]), Point(i * 2, hist_h - yen_f[i]),
			Scalar(0, 0, 255), 2, 5, 0);
	}

	namedWindow("Hist Huang2", WINDOW_AUTOSIZE);
	imshow("Hist Huang2", histImage1);

	namedWindow("Hist Yen", WINDOW_AUTOSIZE);
	imshow("Hist Yen", histImage2);


	/*
	int pos = filename.find(".png");
	string filename_to_huang2 = filename.substr(0, pos) + "_huang2.png";
	string filename_to_yen = filename.substr(0, pos) + "_yen.png";

	imwrite(filename_to_huang2, im_huang2);
	imwrite(filename_to_yen, im_yen);

	/amedWindow("original", WINDOW_AUTOSIZE);
	imshow("original", im);

	namedWindow("huagn2", WINDOW_AUTOSIZE);
	imshow("huagn2", im_huang2);

	namedWindow("yen", WINDOW_AUTOSIZE);
	imshow("yen", im_yen);*/
	
	waitKey(0);
	return 0;
}