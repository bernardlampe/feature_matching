#ifndef _SUPPORT_H_
#define _SUPPORT_H_

/* read the points */
void readPtsFile(const char *fname, vector<CvPoint3D64f> &pts)
{
  ifstream is;
  int numPts;
  double x, y;

  // clear return vector
  pts.clear();

  is.open(fname);
  if (!is.good()) throw Exception("could not open pts file");

  // first line is number of points
  is >> numPts;

  // read points in "col row" format and store as (x, y, 1.0)
  for(int i = 0; i < numPts; i++) { is >> x >> y; pts.push_back(cvPoint3D64f(x, y, 1.0)); }

  is.close();
}

/* print matrix to standard out for analysis */
void PrintMat(CvMat *Fp)
{
  for(int i = 0; i < Fp->height; i++)
  {
    for(int j = 0; j < Fp->width; j++)
    {
      cout << setw(15) << setprecision(5) << cvmGet(Fp, i, j) << " ";
    }
    cout << endl;
  }

  cout << endl;
}

/* draw a cross on the image */
void drawCross(IplImage *img, CvPoint3D64f &p, CvScalar &color, int l, int thick)
{
  cvLine(img, cvPoint(p.x - l, p.y), cvPoint(p.x + l, p.y), color, thick);
  cvLine(img, cvPoint(p.x, p.y - l), cvPoint(p.x, p.y + l), color, thick);
}

/* draw a circle on the image */
void drawCircle(IplImage *img, CvPoint3D64f &p, CvScalar &color, int d, int thick)
{
  cvCircle(img, cvPoint(p.x, p.y), d, color, thick);
}

#endif // _SUPPORT_H_

