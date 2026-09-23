#ifndef _ERROR_H_
#define _ERROR_H_

/* Compute two-way error.
   error = sqrt( d(x, H^-1 * x')^2 + d(x', H * x)^2 ) */
double computeError(
     const CvMat *H, // estimate homography
     const double &t, // distance threshold
     const vector<CvPoint3D64f> &pts1, // set of image #1 points
     const vector<CvPoint3D64f> &pts2, // set of image #2 points
     vector<int> &inlierInds, // set of pts indicies of inliers
     vector<int> &outlierInds) // set of pts indicies of outliers
{
  // clear returning vectors
  inlierInds.clear();
  outlierInds.clear();

  // compute inverse of H
  CvMat *Hinv = cvCreateMat(3, 3, CV_64FC1);
  cvInvert(H, Hinv);

  // compute two-way error
  CvPoint3D64f p;
  double e, error = 0;
  for(unsigned i = 0; i < pts1.size(); i++)
  {
    // compute p = H * x
    p.x = cvmGet(H, 0, 0) * pts1[i].x + cvmGet(H, 0, 1) * pts1[i].y + cvmGet(H, 0, 2) * pts1[i].z;
    p.y = cvmGet(H, 1, 0) * pts1[i].x + cvmGet(H, 1, 1) * pts1[i].y + cvmGet(H, 1, 2) * pts1[i].z;
    p.z = cvmGet(H, 2, 0) * pts1[i].x + cvmGet(H, 2, 1) * pts1[i].y + cvmGet(H, 2, 2) * pts1[i].z;

    // perspective projection
    p.x /= p.z; p.y /= p.z; p.z /= p.z;

    // p = p - x'
    p.x -= pts2[i].x; p.y -= pts2[i].y; p.z -= pts2[i].z;

    // compute ||p - x'||^2
    double m1 = p.x * p.x + p.y * p.y + p.z * p.z;

    // compute p = H^-1 * x'
    p.x = cvmGet(Hinv, 0, 0) * pts2[i].x + cvmGet(Hinv, 0, 1) * pts2[i].y + cvmGet(Hinv, 0, 2) * pts2[i].z;
    p.y = cvmGet(Hinv, 1, 0) * pts2[i].x + cvmGet(Hinv, 1, 1) * pts2[i].y + cvmGet(Hinv, 1, 2) * pts2[i].z;
    p.z = cvmGet(Hinv, 2, 0) * pts2[i].x + cvmGet(Hinv, 2, 1) * pts2[i].y + cvmGet(Hinv, 2, 2) * pts2[i].z;

    // perspective projection
    p.x /= p.z; p.y /= p.z; p.z /= p.z;

    // p = p - x'
    p.x -= pts1[i].x; p.y -= pts1[i].y; p.z -= pts1[i].z;

    // compute ||p - x||^2
    double m2 = p.x * p.x + p.y * p.y + p.z * p.z;

    // error is sqrt of d(x, H^-1 * x')^2 + d(x', H * x)^2
    e = sqrt(m1 + m2);

    // compute inlier/outlier sets
    if (e < t)
    {
      inlierInds.push_back(i);

      // accumulate fitting error
      error += m1 + m2;
    }
    else
    {
      outlierInds.push_back(i);
    }
  }

  // release memory
  cvReleaseMat(&Hinv);

  // fitting error = sqrt( sum( |x2 - H*x1|^2 + |x1 - H^-1|^2 ) );
  return(sqrt(error));
}

#endif // _ERROR_H_
