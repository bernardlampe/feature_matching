/* Bernard Lampe
   EECS 442, Computer Vision
   University of Michigan
   Homework #3 */

/* This following code will automatically find
   the outliers in the set of image correspondences.
   It estimates the Homography using RANSAC.
   It utilizes the opencv 2.0.0 API */

/* Usage:
   ./main
     arg 1 = 2D points text file for first image
     arg 2 = first image file
     arg 3 = 2D points text file for second image
     arg 4 = second image file
     arg 5 = distance threshold */

#include<exception>
#include<iomanip>
#include<fstream>
#include<limits>
#include<iostream>
#include<vector>

using namespace std;

#include "cv.h"
#include "highgui.h"
#include "Exception.h"
#include "Error.h"
#include "Support.h"
#include "Normalize.h"
#include "Homography.h"
#include "Reject_Outliers.h"

int main(int argc, char **argv)
{
  if (argc != 6)
  {
    cerr << argv[0] << " <ptsFile_img1.txt> <img1>"
                    << " <ptsFile_img2.txt> <img2>"
                    << " <distThresh>" << endl;
    return(1);
  }

  try
  {
    // read the corresponding points files
    vector<CvPoint3D64f> pts1, pts2;
    readPtsFile(argv[1], pts1);
    readPtsFile(argv[3], pts2);

    // initialize the random number generator
    srand(time(NULL));

    // set distance threshold
    double distThresh = atof(argv[5]);

    // compute inlier and outlier sets using RANSAC
    vector<int> inlierInds, outlierInds;
    Reject_Outliers(pts1, pts2, distThresh, inlierInds, outlierInds);

    // read image files
    IplImage *img1 = cvLoadImage(argv[2]);
    IplImage *img2 = cvLoadImage(argv[4]);
    if (!img1 || !img2) throw Exception("could not load image files");

    // set colors
    CvScalar red = cvScalar(0, 0, 255);
    CvScalar green = cvScalar(0, 255, 0);

    // annotate images with inlier set
    for(unsigned i = 0; i < inlierInds.size(); i++)    
    {
      // draw green cross marks
      drawCross(img1, pts1[inlierInds[i]], green, 5, 2);
      drawCross(img2, pts2[inlierInds[i]], green, 5, 2);
    }

    // write annotated images
    cvSaveImage("img1_inliers.png", img1);
    cvSaveImage("img2_inliers.png", img2);

    // free memory
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);

    // read image files
    img1 = cvLoadImage(argv[2]);
    img2 = cvLoadImage(argv[4]);
    if (!img1 || !img2) throw Exception("could not load image files");

    // annotate images with outlier set
    for(unsigned i = 0; i < outlierInds.size(); i++)    
    {
      drawCircle(img1, pts1[outlierInds[i]], red, 5, 2);
      drawCircle(img2, pts2[outlierInds[i]], red, 5, 2);
    }

    // write annotated images
    cvSaveImage("img1_outliers.png", img1);
    cvSaveImage("img2_outliers.png", img2);

    // free memory
    cvReleaseImage(&img1);
    cvReleaseImage(&img2);
  }
  catch(Exception &e)
  {
    cerr << "Error: " << e.what() << endl;
    return(1);
  }
  catch(...)
  {
    cerr << "caught unhandled exception" << endl;
    return(1);
  }

  return(0);
}

