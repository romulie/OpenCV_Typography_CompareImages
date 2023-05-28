// Opens 2 images and compares them taking into account relative shift between them
// use several trackbars to dynamically select regions of interest, template regions (to find match),
// thresholds, minimum defect area etc.
// Addition: uses Compare Histogram methods to evaluate similarity (outputs to console)
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include <iostream>
#include <vector>

const std::string filename_first = "ref.tiff"; // reference image
const std::string filename_second = "ref_withDefect.tif"; // image to be compared with reference
// points and coordinates to be updated dynamically using trackbars
cv::Point ref_TopLeft(20, 100), ref_BotRight(1200, 900);       // margins for reference image
cv::Point templ_TopLeft(1085, 100), templ_BotRight(1185, 400); // margins for template region
int ref_corner_x = 20, templ_corner_x = 1085; // same but as separate coordinates and dimensions 
int ref_corner_y = 100, templ_corner_y = 100;
int ref_width = 1200, templ_width = 100;
int ref_height = 900, templ_height = 300;

cv::Rect ref_crop_rect(20, 100, 1200, 900); // offset x,y (to accomodate shift during matching); size x,y 
cv::Rect templ_crop_rect(1085, 100, 100, 300); // template crop rectangle (should have prominent image feature, should be long along y-axis since y-shift is bigger than x-shift )

int threshold_value = 50;        // minimum pixel difference value for thresholding
int max_threshold_value = 255;   // pixel value set after thresholding
int errode_dilate_seed = 0;       
int threshold_defect_area = 300;
cv::RNG rng(12345);              // random generator for color contours visualization

void show(const std::string& name, const cv::Mat& img, int xSize = 500, int ySize = 400, int xOffset = 0, int yOffset = 0) {
    cv::namedWindow(name, 0);
    cv::resizeWindow(name, xSize, ySize);
    cv::moveWindow(name, xOffset, yOffset);
    cv::imshow(name, img);
}
cv::Mat image_first, image_second, copy1, copy2;

void histogramComparison(cv::Mat img1, cv::Mat img2) {
    int histSize = 256;
    float range[] = { 0, 256 }; //the upper boundary is exclusive
    const float* histRange[] = { range };
    bool uniform = true, accumulate = false;
    cv::Mat img1_hist,img2_hist; // histograms of images
    calcHist(&img1, 1, 0, cv::Mat(), img1_hist, 1, &histSize, histRange, uniform, accumulate);
    calcHist(&img2, 1, 0, cv::Mat(), img2_hist, 1, &histSize, histRange, uniform, accumulate);
    
    int hist_w = 512, hist_h = 400;
    cv::normalize(img1_hist, img1_hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());
    cv::normalize(img2_hist, img2_hist, 0, hist_h, cv::NORM_MINMAX, -1, cv::Mat());

    // Draw histograms (no purpose, just for visualisation) // Ñomment for speeding-up
    int bin_w = cvRound((double)hist_w / histSize);
    cv::Mat histImage1(hist_h, hist_w, CV_8UC3, cv::Scalar(0));
    cv::Mat histImage2(hist_h, hist_w, CV_8UC3, cv::Scalar(0));
    for (int i = 1; i < histSize; i++) {
        cv::line(histImage1, cv::Point(bin_w * (i - 1), hist_h - cvRound(img1_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(img1_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
        cv::line(histImage2, cv::Point(bin_w * (i - 1), hist_h - cvRound(img2_hist.at<float>(i - 1))),
            cv::Point(bin_w * (i), hist_h - cvRound(img2_hist.at<float>(i))),
            cv::Scalar(255, 0, 0), 2, 8, 0);
    }
    show("Hist of img1", histImage1, 200, 200, 1100, 0);    //cv::imshow("Hist of img1", histImage1);
    show("Hist of img2", histImage2, 200, 200, 1100, 300);  // cv::imshow("Hist of img2", histImage2)
    // End of Draw histograms 
    
    // Compare histograms using different methods (Correlation(0)/Chi-Square(1)/Intersection(2)/Bhattacharyya distance(3))
    for (int compare_method = 0; compare_method < 4; compare_method++){
        double hist_compare_val = cv::compareHist(img1_hist, img2_hist, compare_method);
        std::cout << "Method " << compare_method << "  hist_compare_val = " << hist_compare_val << std::endl;
    }
    std::cout << std::endl;
}

void doWork(int, void*) {
    // update regions
    ref_TopLeft.x = ref_corner_x; ref_TopLeft.y = ref_corner_y;
    ref_BotRight.x = ref_TopLeft.x + ref_width; ref_BotRight.y = ref_TopLeft.y + ref_height;
    cv::rectangle(image_first, ref_TopLeft, ref_BotRight, cv::Scalar(255, 0, 0), 2, cv::LINE_8); // reference rectangle
    templ_TopLeft.x = templ_corner_x; templ_TopLeft.y = templ_corner_y;
    templ_BotRight.x = templ_TopLeft.x + templ_width; templ_BotRight.y = templ_TopLeft.y + templ_height;
    cv::rectangle(image_first, templ_TopLeft, templ_BotRight, cv::Scalar(255, 0, 0), 8, cv::LINE_8); // template rectangle
    show(filename_first, image_first);
    copy1.copyTo(image_first); // restore image after drawing selection rectangles
    
    // update template crop rectangle
    templ_crop_rect.x = templ_corner_x; templ_crop_rect.y = templ_corner_y;
    templ_crop_rect.width = templ_width; templ_crop_rect.height = templ_height;
    // Make template for searching match according to the templ_crop_rect
    cv::Mat image_template = image_first(templ_crop_rect);
    cv::Mat image_match_result = cv::Mat::zeros(image_second.size(), image_second.type());
    // find match and best match position
    cv::matchTemplate(image_second, image_template, image_match_result, cv::TM_CCORR_NORMED);
    double minVal, maxVal;
    cv::Point minLoc, maxLoc, template_delta_point;
    cv::minMaxLoc(image_match_result, &minVal, &maxVal, &minLoc, &maxLoc);
    template_delta_point = cv::Point(templ_crop_rect.x, templ_crop_rect.y) - maxLoc;
    std::cout << "MAXIMUM MATCH LOCATION = " << maxLoc << ";  DELTA = " << template_delta_point << std::endl;
    /// update reference crop rectangle
    ref_crop_rect.x = ref_corner_x; ref_crop_rect.y = ref_corner_y;
    ref_crop_rect.width = ref_width; ref_crop_rect.height = ref_height;
    // crop image to match according to the match found
    const cv::Rect image_crop_rect(ref_crop_rect.x - template_delta_point.x, ref_crop_rect.y - template_delta_point.y,
        ref_crop_rect.width, ref_crop_rect.height);
    std::cout << "Image_crop_rect = " << image_crop_rect << std::endl;
    // draw found matching rectangles for/on the second image 
    cv::rectangle(image_second, ref_TopLeft - template_delta_point, ref_BotRight - template_delta_point, cv::Scalar(255, 0, 0), 2, cv::LINE_8); // found reference rectangle
    cv::rectangle(image_second, maxLoc, cv::Point(maxLoc.x + templ_width, maxLoc.y + templ_height), cv::Scalar(255, 0, 0), 8, cv::LINE_8); // found template rectangle
    show(filename_second, image_second, 500, 400, 550, 0);
    copy2.copyTo(image_second); // restore image after drawing matching rectangles

    // crop two images to the same size for comparison and according to the matchTemplate found result
    cv::Mat img1_croped = image_first(ref_crop_rect);
    cv::Mat img2_croped = image_second(image_crop_rect);
 // Use Histogram Comparison to perform a comparison
    histogramComparison(img1_croped, img2_croped);

 // Find absolute difference between two images
    cv::Mat image_diff = cv::Mat::zeros(img1_croped.size(), img1_croped.type());
    cv::absdiff(img1_croped, img2_croped, image_diff);
    cv::Mat image_diff_copy; image_diff.copyTo(image_diff_copy); // copy for restoration between iterations
    // make THRESHOLDING in order to compensate for exposure differences
    cv::threshold(image_diff, image_diff, threshold_value, max_threshold_value, 0); // 0 = Binary Type Thresholding
    // ? errode - dilate ?
    int er_dil_type = cv::MORPH_RECT;
    cv::Mat element = cv::getStructuringElement(er_dil_type, cv::Size(2 * errode_dilate_seed + 1, 2 * errode_dilate_seed + 1), cv::Point(errode_dilate_seed, errode_dilate_seed));
    erode(image_diff, image_diff, element);
    dilate(image_diff, image_diff, element);
    // find count of nonzero pixels in difference image
    int nonZeroCount = cv::countNonZero(image_diff);
    std::cout << "Discrepancy Pixels Count = " << nonZeroCount << std::endl;

    // Find continuous contours.
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(image_diff, contours, hierarchy, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
    // Draw colourful borders around detected contours GREATER THAN A PARTICULAR MINIMUM SIZE
    cv::Mat image_diff_contours = image_diff.clone();
    cv::merge(std::vector<cv::Mat>{image_diff_contours, image_diff_contours, image_diff_contours}, image_diff_contours);
    int contours_cnt = 0;
    for (size_t i = 0; i < contours.size(); ++i) {
        cv::Scalar random_color = cv::Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        if (cv::contourArea(contours[i]) > threshold_defect_area) {
            ++contours_cnt;
            cv::drawContours(image_diff_contours, contours, (int)i, random_color, 6, cv::LINE_8, hierarchy, 0);
        }
    }
    std::cout << "Count of regions with area > threshold_defect_area = " << contours_cnt << std::endl;
    show("DIFFERENCE", image_diff_contours, 500, 400, 550, 200);
    image_diff_copy.copyTo(image_diff); // restore difference image after displaying
}
int main()
{   // Load two input images to compare
    image_first = cv::imread(filename_first, cv::IMREAD_GRAYSCALE); //cv::IMREAD_COLOR
    if (image_first.empty()) std::cout << "Couldn't load " << filename_first << std::endl;
    image_second = cv::imread(filename_second, cv::IMREAD_GRAYSCALE);
    if (image_second.empty()) std::cout << "Couldn't load " << filename_second << std::endl;
    image_first.copyTo(copy1); // copy of first image to restore after drawing selection rectangles
    image_second.copyTo(copy2); // copy of second image to restore after drawing matching rectangles

    cv::namedWindow(filename_first, 0); // first(reference) image window and trackbars for region-of-search selection
    cv::createTrackbar("r_x_TL ", filename_first, &ref_corner_x, 1400, doWork);
    cv::createTrackbar("r_y_TL ", filename_first, &ref_corner_y, 1000, doWork);
    cv::createTrackbar("r_x_BR ", filename_first, &ref_width, 1400, doWork);
    cv::createTrackbar("r_y_BR ", filename_first, &ref_height, 1000, doWork);
    cv::namedWindow(filename_second, 0);// second image window and trackbars for template position and size selection
    cv::createTrackbar("t_x_TL ", filename_second, &templ_corner_x, 1400, doWork);
    cv::createTrackbar("t_y_TL ", filename_second, &templ_corner_y, 1000, doWork);
    cv::createTrackbar("t_x_BR ", filename_second, &templ_width, 1400, doWork);
    cv::createTrackbar("t_y_BR ", filename_second, &templ_height, 1000, doWork);
    cv::namedWindow("DIFFERENCE", 0); // difference image window and trackbars for thresholding, errode/dilate processing and minimum detectable defect area
    cv::createTrackbar("thrshld ", "DIFFERENCE", &threshold_value, 255, doWork);
    cv::createTrackbar("err/dil ", "DIFFERENCE", &errode_dilate_seed, 50, doWork);
    cv::createTrackbar("min.area ", "DIFFERENCE", &threshold_defect_area, 2000, doWork);
    doWork(0, 0);

    int c = cv::waitKey();
    if (c == 27) return 0;

    return 0;
}