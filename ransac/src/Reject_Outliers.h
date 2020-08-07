#ifndef _REJECT_OUTLIERS_H_
#define _REJECT_OUTLIERS_H_

/* Use RANSAC to find the set of outliers.
   Implementation from "Multiple View Geometry in Computer Vision",
   Hartley and Zisserman.  Algorithm 4.6. */
void Reject_Outliers(
     const vector<CvPoint3D64f> &pts1, // image #1 point set
     const vector<CvPoint3D64f> &pts2, // image #2 point set
     const double &distThresh, // distance threshold
     vector<int> &inlierInds,  // return set of inlier indicies for point set #1
     vector<int> &outlierInds) // return set of outliers indicies for point set #2
{
  // check args
  if (pts1.size() != pts2.size()) throw(Exception("vector arrays must have equal dimensions"));
  if (pts1.size() < 4) throw(Exception("must specify >= 4 points to compute Homography"));

  // clear return vectors
  inlierInds.clear();
  outlierInds.clear();

  // get number of correspondences
  int numPts = pts1.size();

  // initialize high watermark and set optError = inf.
  unsigned optNumInliers = 0;
  double optError = numeric_limits<double>::max();

  // compute number of iterations to ensure sample set
  // of only inliers is chosen at probability = 95%
  int N = int(ceil(log(1.0 - 0.95) / log(1.0 - pow(0.83, 4.0)))); // assume 50 inliers to 10 outliers

  // iterate and find consensus set
  for(int i = 0; i < N; i++)
  {
    // choose a random sample of size 4
    vector<CvPoint3D64f> pts1p, pts2p;
    for(int j = 0; j < 4; j++)
    {
      int ind = rand() % numPts;
      pts1p.push_back(pts1[ind]);
      pts2p.push_back(pts2[ind]);
    }

    // compute Homography
    CvMat *H = cvCreateMat(3, 3, CV_64FC1);
    Homography(pts1p, pts2p, H);

    // compute two-way error
    vector<int> inliers, outliers;
    double error = computeError(H, distThresh, pts1, pts2, inliers, outliers);

    // print out the stats for this iteration
    cout << "------------------------------" << endl;
    cout << "Iteration #: " << i << endl;
    cout << "Number of Outliers: " << outliers.size() << endl;
    cout << "Fitting Error: " << error << endl;

    // keep model with largest number of inliers and min error
    if (inliers.size() >= optNumInliers && error < optError)
    {
      // print status
      cout << "***** Found outlier set with minimum cardinality" << endl;

      // keep the best model
      optNumInliers = inliers.size();
      optError = error;

      // keep track of best out/in liers sets
      outlierInds = outliers;
      inlierInds = inliers;
    }

    // release memory
    cvReleaseMat(&H);
  }

  // compute optimal homography from inlier set
  CvMat *optH = cvCreateMat(3, 3, CV_64FC1);
  vector<CvPoint3D64f> pts1p, pts2p;
  for(unsigned i = 0; i < inlierInds.size(); i++)
  { 
    int ind = inlierInds[i];
    pts1p.push_back(pts1[ind]);
    pts2p.push_back(pts2[ind]);
  }
  Homography(pts1p, pts2p, optH);
  
  // print out best fit homography
  cout << endl << "Optimal Homography:" << endl;
  cout << "------------------------------" << endl;
  PrintMat(optH);

  // print out summary of best fit
  cout << "Summary:" << endl;
  cout << "------------------------------" << endl;
  cout << "Number of Inliers: " << inlierInds.size() << endl;
  cout << "Number of Outliers: " << outlierInds.size() << endl;
  cout << "Fitting Error: " << optError << endl;

  // release memory
  cvReleaseMat(&optH);
}

#endif // _REJECT_OUTLIERS_H_
