#ifndef _NORMALIZE_H_
#define _NORMALIZE_H_

/* Normalize points so that origin is points centroid and
   the mean distance from the origin is sqrt(2).
   Return the similarity transform */
CvMat* normalizePts(
     vector<CvPoint3D64f> &pts) // set of points to normalize
{
  // get centroid of points
  double c_x = 0;
  double c_y = 0;
  for(unsigned i = 0; i < pts.size(); i++) { c_x += pts[i].x; c_y += pts[i].y; }
  c_x /= double(pts.size()); c_y /= double(pts.size());

  // translate points so origin is points centroids
  for(unsigned i = 0; i < pts.size(); i++) { pts[i].x -= c_x; pts[i].y -= c_y; }

  // get the mean distance
  double dist = 0;
  for(unsigned i = 0; i < pts.size(); i++) { dist += sqrt(pts[i].x * pts[i].x + pts[i].y * pts[i].y);  }
  dist /= double(pts.size());

  // compute scale factor
  double scale = sqrt(2.0) / dist;

  // scale points
  for(unsigned i = 0; i < pts.size(); i++) { pts[i].x *= scale; pts[i].y *= scale; }

  // construct similarity transform
  CvMat *T = cvCreateMat(3, 3, CV_64FC1);
  cvSetIdentity(T);
  cvmSet(T, 0, 0, scale);
  cvmSet(T, 1, 1, scale);
  cvmSet(T, 0, 2, -scale * c_x);
  cvmSet(T, 1, 2, -scale * c_y);

  return(T);
}

#endif // _NORMALIZE_H_
