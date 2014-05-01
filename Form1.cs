using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Text;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;
using Emgu.CV.VideoSurveillance;
using Emgu.CV.Features2D;
using Emgu.Util;


namespace Multiplicity
{
   
   public partial class Form1 : Form
   {

      OpenFileDialog OpenImages = new OpenFileDialog();

      float WhiteCorrelation(Image<Gray, Byte> fimage, Image<Gray, Byte> limage)
      {
          int totalinl = 0;
          int totalmatch = 0;
          for (int i = 0; i < fimage.Height; i++)
          {
              for (int j = 0; j < limage.Width; j++)
              {
                  if (limage.Data[i,j,0] == 255)
                  {
                      totalinl++;
                      if(fimage.Data[i, j, 0] == limage.Data[i, j, 0])  {
                        totalmatch++;
                      }
                  }
              }
          }
          return (float)totalmatch/(float)totalinl;
      }

      Image<Bgr, float> BackgroundFromImages(Image<Bgr, float> fImage, Image<Bgr, float> lImage)
      {
          Image<Bgr, float> output = new Image<Bgr, float>(fImage.Size);
          Image<Gray, Byte> fImage_gray = fImage.Convert<Gray, Byte>();
          Image<Gray, Byte> lImage_gray = lImage.Convert<Gray, Byte>();

          // HSV
          Image<Hsv, float> hsv_fImage = fImage.Convert<Hsv, float>();
          Image<Hsv, float> hsv_lImage = lImage.Convert<Hsv, float>(); 
          // Compute HSV Difference 
          Image<Hsv, float> hsv_diff = new Image<Hsv, float>(fImage.Size);
          Image<Gray, Byte> gdiff = new Image<Gray, Byte>(fImage.Size); // Gray RGB difference
          Emgu.CV.CvInvoke.cvAbsDiff(fImage_gray, lImage_gray, gdiff);
          Emgu.CV.CvInvoke.cvAbsDiff(hsv_fImage, hsv_lImage, hsv_diff);
          Image<Gray, float> gray_diff = hsv_diff.Convert<Gray, float>();

          Image<Bgr, float> background = gdiff.Convert<Bgr, float>();
          Image<Gray, Byte> background_gray = background.Convert<Gray,Byte>();
          Image<Gray, float> binary_mask = new Image<Gray, float>(background.Size);

          // Now find which segmented foreground is from which image
          // Create a binary mask of the foregrounds
          //CvInvoke.cvShowImage("bmassskd", background_gray);
          background_gray.Save("D:\\bmassskd.jpg");
          float threshold = 35;
          for (int i = 0; i < background_gray.Height; i++)
          {
              for (int j = 0; j < background_gray.Width; j++)
              {
                  if (background_gray.Data[i, j, 0] > threshold)
                  {
                      binary_mask.Data[i, j, 0] = 255;
                  }
                  else
                  {
                      binary_mask.Data[i, j, 0] = 0;
                  }

              }
          }


          // Enhance the binary mask
          int n = 15;
          IntPtr circle = Emgu.CV.CvInvoke.cvCreateStructuringElementEx(n, n, n / 2, n / 2, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_ELLIPSE, System.IntPtr.Zero);
          Image<Gray, float> binary_mask_e = new Image<Gray, float>(background_gray.Size);
          Image<Gray, float> temp = new Image<Gray, float>(background_gray.Size);
          CvInvoke.cvMorphologyEx(binary_mask, binary_mask_e, temp, circle, Emgu.CV.CvEnum.CV_MORPH_OP.CV_MOP_OPEN, 1);
          Emgu.CV.CvInvoke.cvDilate(binary_mask_e, binary_mask_e, circle,3);

          
          Image<Gray, Byte> binarymask = binary_mask_e.Convert<Gray, Byte>();
          
          for (int i = 0; i < binarymask.Height; i++)
          {
              for (int j = 0; j < 6; j++)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
          for (int i = 0; i < binarymask.Height; i++)
          {
              for (int j = binarymask.Width - 1; j > binarymask.Width - 7; j--)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
          for (int i = 0; i < 6; i++)
          {
              for (int j = 0; j < binarymask.Width; j++)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
          for (int i = binarymask.Height - 1; i > binarymask.Height - 7; i--)
          {
              for (int j = 0; j < binarymask.Width; j++)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
         // CvInvoke.cvShowImage("bmasssk", binarymask);
          binarymask.Save("D:\\bmasssk.jpg");
          // We apply canny edge detector to each image
          Image<Gray, Byte> fEdges = new Image<Gray, Byte>(fImage.Size);
          CvInvoke.cvCanny(fImage_gray, fEdges, 10, 60, 3);
          Image<Gray, Byte> lEdges = new Image<Gray, Byte>(lImage.Size);
          CvInvoke.cvCanny(lImage_gray, lEdges, 10, 60, 3);
          Image<Gray, Byte> bEdges = new Image<Gray, Byte>(background.Size);
          CvInvoke.cvCanny(binarymask, bEdges, 10, 60, 3);
          

          MemStorage storage = new MemStorage();          

          
          Rgb r = new Rgb(0, 0, 0);
          Rgb r1 = new Rgb(255, 255, 255);
          Point p = new Point(0, 0);
          MCvScalar color1 = r.MCvScalar;
          MCvScalar color2 = r1.MCvScalar;

          Image<Bgr, float> outputz = new Image<Bgr, float>(fImage.Size);
          int counter = 1;

          for (Contour<Point> contours5 = bEdges.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage); contours5 != null; contours5 = contours5.HNext)
          {
              Contour<Point> currentContour = contours5.ApproxPoly(contours5.Perimeter * 0.000000000005, storage);
              Emgu.CV.CvInvoke.cvDrawContours(outputz, currentContour, color2, color1, 0, -1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, p);
          }
          
          MemStorage sstorage = new MemStorage();
          
          Image<Gray, Byte> outputz_gray = outputz.Convert<Gray, Byte>();
          Image<Gray, Byte> oEdges = new Image<Gray, Byte>(outputz.Size);
          CvInvoke.cvCanny(outputz_gray, oEdges, 10, 60, 3);

          IntPtr cross = Emgu.CV.CvInvoke.cvCreateStructuringElementEx(n, n, n / 2, n / 2, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_ELLIPSE, System.IntPtr.Zero);
          CvInvoke.cvMorphologyEx(outputz_gray, outputz_gray, temp, cross, Emgu.CV.CvEnum.CV_MORPH_OP.CV_MOP_CLOSE, 1);
         


          Image<Bgr, float> extracted_background = fImage;
          Image<Gray, float> finaloutput = new Image<Gray, float>(outputz.Size);


          for (Contour<Point> contours5 = outputz_gray.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, sstorage); contours5 != null; contours5 = contours5.HNext)
          {
              if (counter >= 10 )
              {
                 // break;
              }
              Contour<Point> currentContour = contours5.ApproxPoly(contours5.Perimeter * 0.000000000005, sstorage);
              Image<Gray, Byte> currentContourImage = new Image<Gray, Byte>(outputz.Size);
              Emgu.CV.CvInvoke.cvDrawContours(currentContourImage, currentContour, color2, color2, 0, -1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, p);
              


              float fWhiteCorrelation = WhiteCorrelation(fEdges, currentContourImage);
              float lWhiteCorrelation = WhiteCorrelation(lEdges, currentContourImage);

              counter++;

              //CvInvoke.cvShowImage("fEdges", fEdges);
              fEdges.Save("D:\\fEdges.jpg");
           //  CvInvoke.cvShowImage("lEdges", lEdges);
             lEdges.Save("D:\\lEdges.jpg");
           //  CvInvoke.cvShowImage("bg"+fWhiteCorrelation + " " + lWhiteCorrelation+ " " + counter+ " ", currentContourImage);
              
              Console.WriteLine(fWhiteCorrelation);
              Console.WriteLine(lWhiteCorrelation);


              if (fWhiteCorrelation > lWhiteCorrelation) // Foreground is from first image
              {
                  for (int i = 0; i < currentContourImage.Height; i++)
                  {
                      for (int j = 0; j < currentContourImage.Width; j++)
                      {
                          if (currentContourImage.Data[i, j, 0] == 255) // This is the position of the foreground
                          {
                              extracted_background.Data[i, j, 0] = lImage.Data[i, j, 0];
                              extracted_background.Data[i, j, 1] = lImage.Data[i, j, 1];
                              extracted_background.Data[i, j, 2] = lImage.Data[i, j, 2];
                          }
                      }
                  }
              }
          
              else // Foreground is from second image
              {
                  for (int i = 0; i < currentContourImage.Height; i++)
                  {
                      for (int j = 0; j < currentContourImage.Width; j++)
                      {
                          if (currentContourImage.Data[i, j, 0] == 255) // This is the position of the foreground
                          {
                              extracted_background.Data[i, j, 0] =  fImage.Data[i, j, 0];
                              extracted_background.Data[i, j, 1] =  fImage.Data[i, j, 1];
                              extracted_background.Data[i, j, 2] =  fImage.Data[i, j, 2];
                          }
                      }
                  }
              }
          }

          return extracted_background;
      }

     


      public void normalize(Image<Bgr, float> fImage, Image<Bgr, float> lImage)
      {
          // Find standard deviation and mean of intensities of each image
          float totalpixels = fImage.Height * fImage.Width;
          float sum_r_1 = 0;
          float sum_r_2 = 0;
          float sum_g_1 = 0;
          float sum_g_2 = 0;
          float sum_b_1 = 0;
          float sum_b_2 = 0;
          for (int i = 0; i < fImage.Height; i++)
          {
              for (int j = 0; j < fImage.Width; j++)
              {
                  sum_b_1 += fImage.Data[i, j, 0];
                  sum_g_1 += fImage.Data[i, j, 1];
                  sum_r_1 += fImage.Data[i, j, 2];

                  sum_b_2 += lImage.Data[i, j, 0];
                  sum_g_2 += lImage.Data[i, j, 1];
                  sum_r_2 += lImage.Data[i, j, 2];
              }
          }


          float mean_r_1 = sum_r_1 / totalpixels;
          float mean_r_2 = sum_r_2 / totalpixels;
          float mean_g_1 = sum_g_1 / totalpixels;
          float mean_g_2 = sum_g_2 / totalpixels;
          float mean_b_1 = sum_b_1 / totalpixels;
          float mean_b_2 = sum_b_2 / totalpixels;

          float numerator_r_1 = 0.0F;
          float numerator_r_2 = 0.0F;
          float numerator_b_1 = 0.0F;
          float numerator_b_2 = 0.0F;
          float numerator_g_1 = 0.0F;
          float numerator_g_2 = 0.0F;

          for (int i = 0; i < fImage.Height; i++)
          {
              for (int j = 0; j < fImage.Width; j++)
              {
                  float temp;
                  temp = fImage.Data[i, j, 0] - mean_b_1;
                  temp *= temp;
                  numerator_b_1 += temp;
                  temp = fImage.Data[i, j, 1] - mean_g_1;
                  temp *= temp;
                  numerator_g_1 += temp;
                  temp = fImage.Data[i, j, 2] - mean_r_1;
                  temp *= temp;
                  numerator_r_1 += temp;

                  temp = lImage.Data[i, j, 0] - mean_b_2;
                  temp *= temp;
                  numerator_b_2 += temp;
                  temp = lImage.Data[i, j, 1] - mean_g_2;
                  temp *= temp;
                  numerator_g_2 += temp;
                  temp = lImage.Data[i, j, 2] - mean_r_2;
                  temp *= temp;
                  numerator_r_2 += temp;
              }
          }

          float sd_r_1 = (float)Math.Sqrt(numerator_r_1 / totalpixels);
          float sd_g_1 = (float)Math.Sqrt(numerator_g_1 / totalpixels);
          float sd_b_1 = (float)Math.Sqrt(numerator_b_1 / totalpixels);

          float sd_r_2 = (float)Math.Sqrt(numerator_r_2 / totalpixels);
          float sd_g_2 = (float)Math.Sqrt(numerator_g_2 / totalpixels);
          float sd_b_2 = (float)Math.Sqrt(numerator_b_2 / totalpixels);

          Image<Bgr, float> normalizedImage = new Image<Bgr, float>(lImage.Size);
          for (int i = 0; i < fImage.Height; i++)
          {
              for (int j = 0; j < fImage.Width; j++)
              {
                  normalizedImage.Data[i, j, 0] = (sd_b_1 / sd_b_2) * (lImage.Data[i,j,0] - mean_b_2) + mean_b_1;
                  normalizedImage.Data[i, j, 1] = (sd_g_1 / sd_g_2) * (lImage.Data[i, j, 1] - mean_g_2) + mean_g_1;
                  normalizedImage.Data[i, j, 2] = (sd_r_1 / sd_r_2) * (lImage.Data[i, j, 2] - mean_r_2) + mean_r_1;
              }
          }
        //  CvInvoke.cvShowImage("normalizedImage", normalizedImage);
      }
     

      public void getForeground(Image<Bgr, float> bg, Image<Bgr, float> foreground, Image<Bgr, float> output)
      {
          Image<Gray, Byte> bg_gray = bg.Convert<Gray, Byte>();
          Image<Gray, Byte> foreground_gray = foreground.Convert<Gray, Byte>();
          // HSV
          Image<Hsv, float> hsv_background = bg_gray.Convert<Hsv, float>();
          Image<Hsv, float> hsv_foreground = foreground_gray.Convert<Hsv, float>();
          // Compute HSV Difference 
          Image<Hsv, float> hsv_diff = new Image<Hsv, float>(bg.Size);
          Image<Gray, Byte> gdiff = new Image<Gray, Byte>(bg.Size); // Gray RGB difference
          Emgu.CV.CvInvoke.cvAbsDiff(bg_gray, foreground_gray, gdiff);
          Emgu.CV.CvInvoke.cvAbsDiff(hsv_background, hsv_foreground, hsv_diff);
          Image<Gray, float> gray_diff = hsv_diff.Convert<Gray, float>();

          Image<Bgr, float> background = gdiff.Convert<Bgr, float>();
          Image<Gray, Byte> background_gray = background.Convert<Gray, Byte>();
          Image<Gray, float> binary_mask = new Image<Gray, float>(background.Size);

          // Now find which segmented foreground is from which image
          // Create a binary mask of the foregrounds

          float threshold = 25;
          for (int i = 0; i < background_gray.Height; i++)
          {
              for (int j = 0; j < background_gray.Width; j++)
              {
                  if (background_gray.Data[i, j, 0] > threshold)
                  {
                      binary_mask.Data[i, j, 0] = 255;
                  }
                  else
                  {
                      binary_mask.Data[i, j, 0] = 0;
                  }
              }
          }
          // Enhance the binary mask
          int n = 15;
          IntPtr circle = Emgu.CV.CvInvoke.cvCreateStructuringElementEx(n, n, n / 2, n / 2, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_ELLIPSE, System.IntPtr.Zero);
          Image<Gray, float> binary_mask_e = new Image<Gray, float>(background_gray.Size);
          Image<Gray, float> temp = new Image<Gray, float>(background_gray.Size);
          CvInvoke.cvMorphologyEx(binary_mask, binary_mask_e, temp, circle, Emgu.CV.CvEnum.CV_MORPH_OP.CV_MOP_OPEN, 1);
          Emgu.CV.CvInvoke.cvDilate(binary_mask_e, binary_mask_e, circle, 5);
          Image<Gray, Byte> binarymask = binary_mask_e.Convert<Gray, Byte>();
          for (int i = 0; i < binarymask.Height; i++)
          {
              for (int j = 0; j < 6; j++)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
          for (int i = 0; i < 6; i++)
          {
              for (int j = 0; j < binarymask.Width; j++)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
          for (int i = 0; i < binarymask.Height; i++)
          {
              for (int j = binarymask.Width - 1; j > binarymask.Width - 7; j--)
              {
                  binarymask.Data[i, j, 0] = 0;
              }
          }
          //CvInvoke.cvShowImage("bmaskf", binarymask);
          // We apply canny edge detector to each image
          Image<Gray, Byte> fEdges = new Image<Gray, Byte>(bg.Size);
          CvInvoke.cvCanny(bg_gray, fEdges, 10, 60, 3);
          Image<Gray, Byte> lEdges = new Image<Gray, Byte>(foreground.Size);
          CvInvoke.cvCanny(foreground_gray, lEdges, 10, 60, 3);
          Image<Gray, Byte> bEdges = new Image<Gray, Byte>(background.Size);
          CvInvoke.cvCanny(binarymask, bEdges, 10, 60, 3);

          MemStorage storage = new MemStorage();


          Rgb r = new Rgb(0, 0, 0);
          Rgb r1 = new Rgb(255, 255, 255);
          Point p = new Point(0, 0);
          MCvScalar color1 = r.MCvScalar;
          MCvScalar color2 = r1.MCvScalar;
          Image<Bgr, float> outputz = new Image<Bgr, float>(bg.Size);
          int counter = 1;

          for (Contour<Point> contours5 = bEdges.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage); contours5 != null; contours5 = contours5.HNext)
          {
              Contour<Point> currentContour = contours5.ApproxPoly(contours5.Perimeter * 0.000000000005, storage);
              Emgu.CV.CvInvoke.cvDrawContours(outputz, currentContour, color2, color1, 0, -1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, p);
          }

          MemStorage sstorage = new MemStorage();

          Image<Gray, Byte> outputz_gray = outputz.Convert<Gray, Byte>();
          Image<Gray, Byte> oEdges = new Image<Gray, Byte>(outputz.Size);
          CvInvoke.cvCanny(outputz_gray, oEdges, 10, 60, 3);

          IntPtr cross = Emgu.CV.CvInvoke.cvCreateStructuringElementEx(n, n, n / 2, n / 2, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_ELLIPSE, System.IntPtr.Zero);
          CvInvoke.cvMorphologyEx(outputz_gray, outputz_gray, temp, cross, Emgu.CV.CvEnum.CV_MORPH_OP.CV_MOP_CLOSE, 1);



          Image<Bgr, float> extracted_background = bg;
          Image<Gray, float> finaloutput = new Image<Gray, float>(outputz.Size);
          

          //CvInvoke.cvShowImage("output", outputz_gray);
          for (Contour<Point> contours5 = outputz_gray.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, sstorage); contours5 != null; contours5 = contours5.HNext)
          {
              if (counter >= 6)
              {
                  break;
              }
              Contour<Point> currentContour = contours5.ApproxPoly(contours5.Perimeter * 0.000000000005, sstorage);
              Image<Gray, Byte> currentContourImage = new Image<Gray, Byte>(outputz.Size);
              Emgu.CV.CvInvoke.cvDrawContours(currentContourImage, currentContour, color2, color2, 0, -1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, p);
              counter++;
              for (int i = 0; i < currentContourImage.Height; i++)
                  {
                 for (int j = 0; j < currentContourImage.Width; j++)
                      {
                          if (currentContourImage.Data[i, j, 0] == 255) // This is the position of the foreground
                          {
                              output.Data[i, j, 0] = foreground.Data[i, j, 0];
                              output.Data[i, j, 1] = foreground.Data[i, j, 1];
                              output.Data[i, j, 2] = foreground.Data[i, j, 2];
                          }
                      }
                  }  
            }
      }

      private void InitializeOpenFileDialog()
      {
          this.OpenImages.Filter =
              "Images (*.BMP;*.JPG;*.GIF)|*.BMP;*.JPG;*.GIF|" +
              "All files (*.*)|*.*";
          this.OpenImages.Multiselect = true;
          this.OpenImages.Title = "My Image Browser";
      }

      private Bitmap rotateImage(Bitmap b, float angle)
      {
          //create a new empty bitmap to hold rotated image
          Bitmap returnBitmap = new Bitmap(b.Width, b.Height);
          //make a graphics object from the empty bitmap
          Graphics g = Graphics.FromImage(returnBitmap);
          //move rotation point to center of image
          g.TranslateTransform((float)b.Width / 2, (float)b.Height / 2);
          //rotate
          
          g.RotateTransform(angle);
          //move image back
          g.TranslateTransform(-(float)b.Width / 2, -(float)b.Height / 2);
          //draw passed in image onto graphics object
          g.DrawImage(b, new Point(0, 0));
          return returnBitmap;
      }


      public Image<Bgr, float> alignment(Image<Bgr, float> fImage, Image<Bgr, float> lImage)
      {
          HomographyMatrix homography = null;
          SURFDetector surfCPU = new SURFDetector(500, false);
          VectorOfKeyPoint modelKeyPoints;
          VectorOfKeyPoint observedKeyPoints;
          Matrix<int> indices;

          Matrix<byte> mask;

          int k = 2;
          double uniquenessThreshold = 0.8;


          Image<Gray, Byte> fImageG = fImage.Convert<Gray, Byte>();
          Image<Gray, Byte> lImageG = lImage.Convert<Gray, Byte>();

          //extract features from the object image
          modelKeyPoints = new VectorOfKeyPoint();
          Matrix<float> modelDescriptors = surfCPU.DetectAndCompute(fImageG, null, modelKeyPoints);


          // extract features from the observed image
          observedKeyPoints = new VectorOfKeyPoint();
          Matrix<float> observedDescriptors = surfCPU.DetectAndCompute(lImageG, null, observedKeyPoints);
          BruteForceMatcher<float> matcher = new BruteForceMatcher<float>(DistanceType.L2);
          matcher.Add(modelDescriptors);

          indices = new Matrix<int>(observedDescriptors.Rows, k);
          using (Matrix<float> dist = new Matrix<float>(observedDescriptors.Rows, k))
          {
              matcher.KnnMatch(observedDescriptors, indices, dist, k, null);
              mask = new Matrix<byte>(dist.Rows, 1);
              mask.SetValue(255);
              Features2DToolbox.VoteForUniqueness(dist, uniquenessThreshold, mask);
          }

          int nonZeroCount = CvInvoke.cvCountNonZero(mask);
          if (nonZeroCount >= 4)
          {
              nonZeroCount = Features2DToolbox.VoteForSizeAndOrientation(modelKeyPoints, observedKeyPoints, indices, mask, 1.5, 20);
              if (nonZeroCount >= 4)
                  homography = Features2DToolbox.GetHomographyMatrixFromMatchedFeatures(modelKeyPoints, observedKeyPoints, indices, mask, 2);
          }
          Image<Bgr, Byte> result = Features2DToolbox.DrawMatches(fImageG, modelKeyPoints, lImageG, observedKeyPoints,
           indices, new Bgr(255, 255, 255), new Bgr(255, 255, 255), mask, Features2DToolbox.KeypointDrawType.DEFAULT);
          if (homography != null)
          {
              //draw a rectangle along the projected model
              Rectangle rect = fImageG.ROI;
              PointF[] pts = new PointF[] { 
               new PointF(rect.Left, rect.Bottom),
               new PointF(rect.Right, rect.Bottom),
               new PointF(rect.Right, rect.Top),
               new PointF(rect.Left, rect.Top)};
              homography.ProjectPoints(pts);

              result.DrawPolyline(Array.ConvertAll<PointF, Point>(pts, Point.Round), true, new Bgr(Color.Red), 5);

              Image<Bgr, byte> mosaic = new Image<Bgr, byte>(fImageG.Width + fImageG.Width, fImageG.Height);
              Image<Bgr, byte> warp_image = mosaic.Clone();
              Image<Bgr, float> result2 = new Image<Bgr, float>(fImage.Size);
              Image<Gray, Byte> result3 = new Image<Gray, Byte>(fImage.Size);
              CvInvoke.cvWarpPerspective(fImage.Ptr, result2, homography.Ptr, (int)INTER.CV_INTER_CUBIC + (int)WARP.CV_WARP_FILL_OUTLIERS, new MCvScalar(0));
              return result2;
          }
          return null;
      }

       /*
      public void OpticalFlowAlign(Image<Bgr, Byte> imgA, Image<Bgr,Byte> imgB)
      {
            Image<Gray,Byte> grayA = imgA.Convert<Gray, Byte>();
            Image<Gray,Byte> grayB = imgB.Convert<Gray,Byte>();
            Image<Gray,Byte> pyrBufferA = new Image<Gray,Byte>(imgA.Size);
            Image<Gray,Byte> pyrBufferB = new Image<Gray,Byte>(imgA.Size);
            
            featuresA = grayA.GoodFeaturesToTrack(100, 0.01, 25, 3)
            grayA.FindCornerSubPix(featuresA, New System.Drawing.Size(10, 10),
                                   New System.Drawing.Size(-1, -1),
                                   New Emgu.CV.Structure.MCvTermCriteria(20, 0.03))
            features = featuresA(0).Length
            Emgu.CV.OpticalFlow.PyrLK(grayA, grayB, pyrBufferA, pyrBufferB, _
                                      featuresA(0), New Size(25, 25), 3, _
                                      New Emgu.CV.Structure.MCvTermCriteria(20, 0.03D),
                                      flags, featuresB(0), status, errors)
            pointsA = New Matrix(Of Single)(features, 2)
            pointsB = New Matrix(Of Single)(features, 2)
            For i As Integer = 0 To features - 1
                pointsA(i, 0) = featuresA(0)(i).X
                pointsA(i, 1) = featuresA(0)(i).Y
                pointsB(i, 0) = featuresB(0)(i).X
                pointsB(i, 1) = featuresB(0)(i).Y
            Next
            Dim Homography As New Matrix(Of Double)(3, 3)
            cvFindHomography(pointsA.Ptr, pointsB.Ptr, Homography, HOMOGRAPHY_METHOD.RANSAC, 1, 0);
      }
       */
      public void abDifference(Image<Bgr, float> fImage, Image<Bgr, float> lImage)
      {
          Image<Bgr, Byte> fImageB = fImage.Convert<Bgr, Byte>();
          Image<Bgr, Byte> lImageB = lImage.Convert<Bgr, Byte>();
          double ContourThresh = 0.003; //stores alpha for thread access
          int Threshold = 60;
          Image<Bgr, Byte> contourImage = new Image<Bgr, Byte>(fImageB.Size);
          Image<Bgr, Byte> Difference = new Image<Bgr, Byte>(fImageB.Size);
          Difference = fImageB.AbsDiff(lImageB); //find the absolute difference 
          /*Play with the value 60 to set a threshold for movement*/
          Difference = Difference.ThresholdBinary(new Bgr(Threshold, Threshold, Threshold), new Bgr(255, 255, 255)); //if value > 60 set to 255, 0 otherwise 
          //DisplayImage(Difference.ToBitmap(), resultbox); //display the absolute difference 

          //Previous_Frame = Frame.Copy(); //copy the frame to act as the previous frame
          Point p = new Point(0, 0);
          #region Draw the contours of difference
          //this is tasken from the ShapeDetection Example
          using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
              //detect the contours and loop through each of them
              for (Contour<Point> contours = Difference.Convert<Gray, Byte>().FindContours(
                    Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE,
                    Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST,
                    storage);
                 contours != null;
                 contours = contours.HNext)
              {
                  //Create a contour for the current variable for us to work with
                  Contour<Point> currentContour = contours.ApproxPoly(contours.Perimeter * 0.00005, storage);

                  //Draw the detected contour on the image
                  if (currentContour.Area > ContourThresh) //only consider contours with area greater than 100 as default then take from form control
                  {
                      contourImage.Draw(currentContour, new Bgr(Color.Red), -1);
                      //Emgu.CV.CvInvoke.cvDrawContours(contourImage, currentContour, new Bgr(Color.Red), new Bgr(Color.Red), 0, -1, Emgu.CV.CvEnum.LINE_TYPE.EIGHT_CONNECTED, p);
              
                  }
              }
          #endregion
        //  CvInvoke.cvShowImage("asdf", contourImage);
          contourImage.Save("D:\\asdf.jpg");
         // DisplayImage(Frame.ToBitmap(), CurrentFrame); //display the image using thread safe call
         // DisplayImage(Previous_Frame.ToBitmap(), PreviousFrame); //display the previous image using thread safe call
 
      }
      public Form1()
      {

          /*
          Image<Bgr, float> original = new Image<Bgr, float>("D:\\orig.png");
          CvInvoke.cvShowImage("output3", original);
          Image<Bgr, float> mask = new Image<Bgr, float>("D:\\origmask.png");
          Image<Bgr, float> result = inpainting(original, mask);
          result = inpainting(result, mask);
          
          CvInvoke.cvShowImage("inpainting", result);
          result.Save("D:\\inpainting.jpg");
          */
         Boolean rotation = false;
         InitializeComponent();
         InitializeOpenFileDialog();
         int fileCount = 0;
         int counter = 0;
         Image<Bgr, float> lastImage = null;
         List<Image<Bgr,float>> imageList = new List<Image<Bgr, float>>();
         DialogResult openedfiles = this.OpenImages.ShowDialog();
         if (openedfiles == System.Windows.Forms.DialogResult.OK)
         {
             fileCount = OpenImages.FileNames.Length;
             if (fileCount >= 2)
             {
                 DialogResult dialogResult = MessageBox.Show("Were the images taken using an iOS Device?", "EXIF Fixes", MessageBoxButtons.YesNo);
                 if (dialogResult == DialogResult.Yes)
                 {
                     rotation = true;
                 }
                 else if (dialogResult == DialogResult.No)
                 {
                     rotation = false;
                 }
                 foreach (String file in OpenImages.FileNames)
                 {
                     
                     Image<Bgr, float> curImage = new Image<Bgr, float>(file);
                     
                     if (counter > 0)
                     {
                         Image<Bgr, float> aligned = alignment(curImage, lastImage);
                         curImage = aligned;
                     }
                     lastImage = curImage.Copy();
                     imageList.Add(curImage);
                     counter++;
                     PictureBox pb = new PictureBox();
                     if (rotation)
                     {
                         float x = 180;
                         Bitmap newbmp = curImage.ToBitmap();
                         Bitmap rec = rotateImage(newbmp, x);
                         pb.Image = rec;
                     }
                     else
                     {
                         pb.Image = curImage.ToBitmap();
                     }
                     pb.Width = 400;
                     pb.Height = 400;
                     pb.SizeMode = PictureBoxSizeMode.Zoom;
                     flowLayoutPanel1.Controls.Add(pb);
                 }
             
                 Image<Bgr, float> fImage = imageList[0].Copy();
                 Image<Bgr, float> lImage = imageList[fileCount - 1].Copy();
                 abDifference(fImage, lImage);
                 /*
                  * Trying to normalize the two images
                 normalize(fImage, lImage);
                 Image<Bgr, Byte> fImage_bg = fImage.Convert<Bgr, Byte>();
                 Image<Bgr, Byte> lImage_bg = lImage.Convert<Bgr, Byte>();
                 fImage_bg._EqualizeHist();
                 fImage_bg._GammaCorrect(1.8d);
                 lImage_bg._EqualizeHist();
                 lImage_bg._GammaCorrect(1.8d);
                 CvInvoke.cvShowImage("fimagehist", fImage_bg);
                 CvInvoke.cvShowImage("limagehist", lImage_bg);
                  */
                 // First we extract the background information
                 Image<Bgr, float> background = BackgroundFromImages(fImage, lImage);
                 if (rotation)
                 {
                     float x = 180;
                     Bitmap newbmp = background.ToBitmap();
                     Bitmap rec = rotateImage(newbmp, x);
                     backgroundbox.Image = rec;
                     rec.Save("D:\\rec.jpg");
                 }
                 else
                 {
                     backgroundbox.Image = background.ToBitmap();
                     background.Save("D:\\background.jpg");
                 }
                 backgroundbox.SizeMode = PictureBoxSizeMode.Zoom;
                 // Then we pick foregrounds from the reference background and add them
                 Image<Bgr, float> compiled = new Image<Bgr, float>(fImage.Size);
                 compiled = background.Copy();
                 for (int i = 0; i < fileCount; i++)
                 {
                     Image<Bgr, float> curImage = imageList[i];
                     getForeground(background, curImage, compiled);
                 }

                 if (rotation)
                 {
                     float x = 180;
                     Bitmap newbmp = compiled.ToBitmap();
                     Bitmap rec = rotateImage(newbmp, x);
                     compiledbox.Image = rec; //compiled.ToBitmap();
                     compiledbox.SizeMode = PictureBoxSizeMode.Zoom;
                     String filePath = System.IO.Path.GetDirectoryName(OpenImages.FileNames[0]);
                     rec.Save(filePath + "/compiled.jpg");
                 }
                 else
                 {
                     compiledbox.Image = compiled.ToBitmap();
                     compiledbox.SizeMode = PictureBoxSizeMode.Zoom;
                     String filePath = System.IO.Path.GetDirectoryName(OpenImages.FileNames[0]);
                     compiled.Save(filePath + "/compiled.jpg");
                 }
               }
          }
      }

      private void flowLayoutPanel1_Paint(object sender, PaintEventArgs e)
      {

      }
   }
}
