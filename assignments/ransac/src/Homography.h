#ifndef _HOMOGRAPHY_H_
#define _HOMOGRAPHY_H_

/* Compute homography between both set of points.
   Point sets passed in are modified. */
void Homography(
     vector<CvPoint3D64f> &pts1, // set of image #1 points
     vector<CvPoint3D64f> &pts2, // set of image #2 points
     CvMat *H) // returned estimated homography
{
  // check args
  if (pts1.size() != pts2.size()) throw(Exception("vector arrays must have equal dimensions"));
  if (pts1.size() < 4) throw(Exception("must specify >= 4 points to compute Homography"));

  // normalize points
  CvMat *T1, *T2;
  T1 = normalizePts(pts1);
  T2 = normalizePts(pts2);

  // compose constraint matrix
  CvMat *A = cvCreateMat(3 * pts1.size(), 9, CV_64FC1);
  cvSetZero(A);
  for(unsigned i = 0; i < pts1.size(); i++)
  {
    // [ 0^T   -w'*x^T   y'*x^T ]
    cvmSet(A, 3 * i, 3, -pts2[i].z * pts1[i].x);
    cvmSet(A, 3 * i, 4, -pts2[i].z * pts1[i].y);
    cvmSet(A, 3 * i, 5, -pts2[i].z * pts1[i].z);
    cvmSet(A, 3 * i, 6, pts2[i].y * pts1[i].x);
    cvmSet(A, 3 * i, 7, pts2[i].y * pts1[i].y);
    cvmSet(A, 3 * i, 8, pts2[i].y * pts1[i].z);

    // [ w'*x^T   0^T   -x'*x^T ]
    cvmSet(A, 3 * i + 1, 0, pts2[i].z * pts1[i].x);
    cvmSet(A, 3 * i + 1, 1, pts2[i].z * pts1[i].y);
    cvmSet(A, 3 * i + 1, 2, pts2[i].z * pts1[i].z);
    cvmSet(A, 3 * i + 1, 6, -pts2[i].x * pts1[i].x);
    cvmSet(A, 3 * i + 1, 7, -pts2[i].x * pts1[i].y);
    cvmSet(A, 3 * i + 1, 8, -pts2[i].x * pts1[i].z);

    // [ -y'*x^T   x'*x^T   0^T ]
    cvmSet(A, 3 * i + 2, 0, -pts2[i].y * pts1[i].x);
    cvmSet(A, 3 * i + 2, 1, -pts2[i].y * pts1[i].y);
    cvmSet(A, 3 * i + 2, 2, -pts2[i].y * pts1[i].z);
    cvmSet(A, 3 * i + 2, 3, pts2[i].x * pts1[i].x);
    cvmSet(A, 3 * i + 2, 4, pts2[i].x * pts1[i].y);
    cvmSet(A, 3 * i + 2, 5, pts2[i].x * pts1[i].z);
  }

  // solve A = U D V^T
  CvMat *U = cvCreateMat(3 * pts1.size(), 9, CV_64FC1);
  CvMat *D = cvCreateMat(9, 9, CV_64FC1);
  CvMat *V = cvCreateMat(9, 9, CV_64FC1);
  cvSVD(A, D, U, V, CV_SVD_U_T | CV_SVD_V_T);

  // construct Hp as last row of V^T
  int j = 0;
  CvMat *Hp = cvCreateMat(3, 3, CV_64FC1);
  for(int i = 0; i < 3; i++)
  {
    cvmSet(Hp, i, 0, cvmGet(V, 8, j++));
    cvmSet(Hp, i, 1, cvmGet(V, 8, j++));
    cvmSet(Hp, i, 2, cvmGet(V, 8, j++));
  }

  // H = T2^-1 * Hp * T1
  CvMat *T2inv = cvCreateMat(3, 3, CV_64FC1);
  CvMat *temp = cvCreateMat(3, 3, CV_64FC1);

  cvInvert(T2, T2inv);
  cvMatMul(T2inv, Hp, temp);
  cvMatMul(temp, T1, H);

  // free memory
  cvReleaseMat(&T1); cvReleaseMat(&T2); cvReleaseMat(&T2inv); cvReleaseMat(&temp);
  cvReleaseMat(&A); cvReleaseMat(&U); cvReleaseMat(&D); cvReleaseMat(&V);
  cvReleaseMat(&Hp);
}

#endif // _HOMOGRAPHY_H_
