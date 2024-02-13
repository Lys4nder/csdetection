#include <opencv2/opencv.hpp>
#include <iostream>
#include <map>
#include <curses.h>

std::string detectShape(int sides) {
    if (sides == 3) return "Triunghi";
    if (sides == 4) return "Patrulater";
    if (sides >= 7 && sides <= 20) return "Cerc";
    return "Necunoscut";
}

std::string detectColor(cv::Mat& hsv, cv::Point point, std::map<std::string, cv::Scalar> colorMap) {
    const int h_bins = 256;
    const int s_bins = 256;
    const int histSize[] = { h_bins, s_bins };
    const float h_ranges[] = { 0, 256 };
    const float s_ranges[] = { 0, 256 };
    const float* ranges[] = { h_ranges, s_ranges };
    const int channels[] = { 0, 1 };

    int roiSize = 100;

    cv::Rect roiRect(point.x - roiSize / 2, point.y - roiSize / 2, roiSize, roiSize);
    roiRect &= cv::Rect(0, 0, hsv.cols, hsv.rows);

    cv::Mat roi = hsv(roiRect);
    cv::MatND hist;
    cv::calcHist(&roi, 1, channels, cv::Mat(), hist, 2, histSize, ranges, true, false);
    cv::normalize(hist, hist, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

    double maxCorrelation = -1;
    std::string detectedColor = "Necunoscut";

    for (const auto& colorPair : colorMap) {
        const std::string& colorName = colorPair.first;
        const cv::Scalar& bgrColor = colorPair.second;

        cv::Scalar bgrColorNoAlpha = cv::Scalar(bgrColor[0], bgrColor[1], bgrColor[2]);

        cv::Mat bgrMat(1, 1, CV_8UC3, bgrColorNoAlpha);
        cv::Mat hsvMat;
        cv::cvtColor(bgrMat, hsvMat, cv::COLOR_BGR2HSV);

        cv::MatND hist_color;
        cv::calcHist(&hsvMat, 1, channels, cv::Mat(), hist_color, 2, histSize, ranges, true, false);
        cv::normalize(hist_color, hist_color, 0, 1, cv::NORM_MINMAX, -1, cv::Mat());

        double correlation = cv::compareHist(hist, hist_color, cv::HISTCMP_CORREL);

        if (correlation > maxCorrelation) {
            maxCorrelation = correlation;
            detectedColor = colorName;
        }
    }

    return detectedColor;
}

int main() {
    cv::VideoCapture cap(0);

    if (!cap.isOpened()) {
        std::cerr << "Eroare la deschiderea camerei video!" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    std::map<std::string, cv::Scalar> colorMap = {
    {"Rosu", cv::Scalar(0, 0, 255)},
    {"Portocaliu", cv::Scalar(0, 165, 255)},
    {"Galben", cv::Scalar(0, 255, 255)},
    {"Verde", cv::Scalar(0, 255, 0)},
    {"Albastru", cv::Scalar(255, 0, 0)},
    {"Alb", cv::Scalar(255, 255, 255)},
    {"Mov", cv::Scalar(128, 0, 128)}
    };

    std::map<std::string, int> colorCount;
    std::map<std::string, int> shapeCount;


    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Frame gol!" << std::endl;
            break;
        }

        cv::Mat blurred;
        cv::GaussianBlur(frame, blurred, cv::Size(5, 5), 0);

        cv::Mat hsv;
        cv::cvtColor(blurred, hsv, cv::COLOR_BGR2HSV);

        cv::Mat mask;
        cv::inRange(hsv, cv::Scalar(0, 100, 100), cv::Scalar(10, 255, 255), mask);
        cv::Mat mask2;
        cv::inRange(hsv, cv::Scalar(160, 100, 100), cv::Scalar(180, 255, 255), mask2);
        cv::bitwise_or(mask, mask2, mask);

        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5)));

        std::vector<std::vector<cv::Point>> contours;
        cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        for (size_t i = 0; i < contours.size(); i++) {
            double area = cv::contourArea(contours[i]);

            if (area > 1000) {
                cv::drawContours(frame, contours, static_cast<int>(i), cv::Scalar(0, 255, 0), 2);

                std::vector<cv::Point> approx;
                cv::approxPolyDP(contours[i], approx, 0.02 * cv::arcLength(contours[i], true), true);

                int sides = static_cast<int>(approx.size());
                std::string shape = detectShape(sides);

                cv::Point point = contours[i][0];
                std::string color = detectColor(hsv, point, colorMap);
                
                colorCount[color]++;
                shapeCount[shape]++;

                system("clear");

                for (const auto& pair : colorCount) {
                    std::cout << pair.first << ": " << pair.second << std::endl;
                }

                for (const auto& pair : shapeCount) {
                    std::cout << pair.first << ": " << pair.second << std::endl;
                }

                cv::putText(frame, shape + ", " + color, contours[i][0], cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 2);
            }
        }

        cv::imshow("Detectie Obiecte", frame);

        if (cv::waitKey(30) == 27) {
            std::cout << "Tasta ESC apasata, iesire..." << std::endl;
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
}