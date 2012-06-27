import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.util.ArrayList;

import javax.imageio.ImageIO;
import javax.swing.ImageIcon;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.xml.crypto.dsig.spec.XSLTTransformParameterSpec;

import com.googlecode.javacv.cpp.opencv_core.CvPoint3D32f;
import com.googlecode.javacv.cpp.opencv_features2d;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_features2d.DMatch;
import com.googlecode.javacv.cpp.opencv_nonfree.SURF;
import com.googlecode.javacv.cpp.opencv_features2d.*;

public class HelloWorld {
	
	
	public static final int xscale = 1200;
	public static final int yscale = 1600;
	
	public static GrowingNeuralGas gas = new GrowingNeuralGas();
    public static void main(String[] args) throws Exception {
    	CvMat map = CvMat.create(1000,1000,CV_32F);
    	int k =0;
    	while (k<3)
    	{
	    	for(int i = 40; i<=50; i+=10)
	    	{
	    		CvMat delta_map = build3dMap(i, i + 10);
	    		CvMat add = CvMat.create(map.rows(), map.cols(), map.depth());
	    		cvAdd(delta_map, map, add,null);
	    		map=add;
	       	}
    		System.out.println(k);
			k++;
		}
    	
    	IplImage map_img = IplImage.create(cvSize(map.cols(), map.rows()), IPL_DEPTH_8U, 1);
   	    cvConvertScaleAbs(map, map_img, 1, 0);
   	    map_img = resize(map_img, 1000, 1000);
   	    showImage(map_img);
   	    
   	    
   	    gas.printNodes();
    	
    }
    public static IplImage openImage(String path) 
    {
    	IplImage ret = null;
    	BufferedImage bi;
		try {
			bi = ImageIO.read(new File(path));
	    	ret = IplImage.createFrom(bi);
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
    	return ret;
    }
    
    public static void showImage(IplImage img)
    {
    	JFrame window= new JFrame();
    	window.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    	window.setSize(img.width(), img.height());
    	BufferedImage pic = img.getBufferedImage();
    	JLabel image =new JLabel(new ImageIcon(pic)); 
    	window.add(image);
    	window.setVisible(true);
    	
    }
    
    public static IplImage resize(IplImage img, int width,int heigth)
    {
    	IplImage rezise = IplImage.create(cvSize(width,heigth),img.depth(),img.nChannels());
    	cvResize(img, rezise);
    	return rezise;
    }
    
    public static IplImage sobel(IplImage img)
    {
		IplImage gray = cvCreateImage( img.cvSize() , IPL_DEPTH_8U, 1 );
		IplImage sobel = cvCreateImage( img.cvSize(), IPL_DEPTH_16S, 1 );
		IplImage out = cvCreateImage( img.cvSize() , img.depth(), img.nChannels() );
		cvCvtColor(img, gray, CV_RGB2GRAY );
		//cvCanny(gray, sobel, 1, 3, 3);
		cvLaplace(gray,sobel,3);
		cvCvtScaleAbs(sobel, out, 1, 0);
		return out;
    }
    
    public static KeyPoint findKeypoints(IplImage image)
    {
    	SURF detector = new SURF(500.0);
    	KeyPoint v1 = new KeyPoint();
    	detector.detect(image, null, v1);
    	return v1;
    }
    
    public static DMatch findMatchingFeature(IplImage image1, IplImage image2,KeyPoint keypoints1,KeyPoint keypoints2, CvMat essential)
    {
    	SURF detector = new SURF(500.0);
    	DescriptorExtractor extractor = detector.getDescriptorExtractor();
    	
    	CvMat descriptors_1 = CvMat.create(1, keypoints1.capacity(), CV_32F);
    	CvMat descriptors_2 = CvMat.create(1, keypoints2.capacity(), CV_32F);
    	
    	extractor.compute( image1, keypoints1, descriptors_1 );
    	extractor.compute( image2, keypoints2, descriptors_2 );
    	
    	FlannBasedMatcher matcher = new FlannBasedMatcher();
   	 	DMatch matches = new DMatch();
   	 	matcher.match(descriptors_1, descriptors_2, matches, null);
   	 	
   	 	
   	 	double max_dist = 0; double min_dist = 100;
   	 	
   	 	//-- Quick calculation of max and min distances between keypoints
   	 	for( int i = 0; i < descriptors_1.rows(); i++ )
   	 	{ 
   	 		double dist = matches.position(i).distance();
   	 		if( dist < min_dist ) min_dist = dist;
   	 		if( dist > max_dist ) max_dist = dist;
   	 	}
   	 	
   	 	DMatch good_matches = new DMatch(matches.capacity());

   	 	matches.position(0);
   	 	
   	 	double t = 100;
   	 	
   	 	int j= 0;
   	 	for(int i = 0; i < descriptors_1.rows(); i++ )
   	 	{ 
   	 		CvMat x1= CvMat.create(1, 3, essential.depth());
	 		x1.put(0,0,keypoints1.position(matches.position(i).queryIdx()).pt_x());
	 		x1.put(0,1,keypoints1.position(matches.position(i).queryIdx()).pt_y());
	 		x1.put(0,2,1);
	 		

   	 		CvMat x2= CvMat.create(3, 1, essential.depth());
	 		x2.put(0,0,keypoints2.position(matches.position(i).trainIdx()).pt_x());
	 		x2.put(1,0,keypoints2.position(matches.position(i).trainIdx()).pt_y());
	 		x2.put(2,0,1);
	 		
	 		CvMat m = CvMat.create(1, 3, essential.depth());
	 		CvMat m2 = CvMat.create(1, 1, essential.depth());
	 		
	 		
	 		cvMatMul(x1, essential, m);
	 		cvMatMul(m, x2, m2);

	 		
   	 		if( m2.get(0,0)*m2.get(0,0) < t )
   	 		{
   	 			good_matches.position(j).queryIdx(matches.position(i).queryIdx());
   	 			good_matches.position(j).trainIdx(matches.position(i).trainIdx());
   	 			good_matches.position(j).distance(matches.position(i).distance());
   	   	 		j++;
   	   	 		
   	 		}
   	 	}

   	 	DMatch good_matches2 = new DMatch(j);

   	 	for(int i = 0; i < j; i++ )
   	 	{

	 			good_matches2.position(i).queryIdx(good_matches.position(i).queryIdx());
	 			good_matches2.position(i).trainIdx(good_matches.position(i).trainIdx());
	 			good_matches2.position(i).distance(good_matches.position(i).distance());
   	 	}
   	 	keypoints1.position(0);
   	 	keypoints2.position(0);
   	 	good_matches.position(0);
   	 	good_matches2.position(0);
   	 	matches.position(0);
   	 	
   	 	//IplImage out = IplImage.create(cvSize(2*image1.width(), image1.height()),image1.depth(),image1.nChannels());
   	 	//opencv_features2d.drawMatches(image1, keypoints1, image2, keypoints2, good_matches2, out, cvScalarAll(-1), cvScalarAll(-1), null, 2);
   	 	   	 	
   	 	//showImage(out);
   	    
   	 	return good_matches2;
    
    }
    
    public static CvMat getRotationMatrix(double angel)
    {
    	CvMat ret = CvMat.create(3, 3,CV_32F);
    	double sin_x = Math.sin(angel);
    	double cos_x = Math.cos(angel);
    	ret.put(0, 0, cos_x);
    	ret.put(0, 1, 0);
    	ret.put(0, 2, sin_x);
    	ret.put(1, 0, 0);
    	ret.put(1, 1, 1);
    	ret.put(1, 2, 0);
    	ret.put(2, 0, -sin_x);
    	ret.put(2, 1, 0);
    	ret.put(2, 2, cos_x);
    	
    	return ret;
    	
    }
    
    public static CvMat getTransitionMatrix(double x,double y,double z)
    {
    	CvMat ret = CvMat.create(3, 1,CV_32F);
    	ret.put(0, 0, x);
    	ret.put(1, 0, y);
    	ret.put(2, 0, z);
    	return ret;
    }
    
    public static CvMat getCameraMatrix(CvMat rotation, CvMat transition)
    {
    	CvMat ret = CvMat.create(3, 4,CV_32F);
    	CvMat ret2 = CvMat.create(3, 4,CV_32F);
    	ret.put(0, 0, rotation.get(0,0));
    	ret.put(0, 1, rotation.get(0,1));
    	ret.put(0, 2, rotation.get(0,2));
    	ret.put(0, 3, transition.get(0,0));
    	ret.put(1, 0, rotation.get(1,0));
    	ret.put(1, 1, rotation.get(1,1));
    	ret.put(1, 2, rotation.get(1,2));
    	ret.put(1, 3, transition.get(1,0));
    	ret.put(2, 0, rotation.get(2,0));
    	ret.put(2, 1, rotation.get(2,1));
    	ret.put(2, 2, rotation.get(2,2));
    	ret.put(2, 3, transition.get(2,0));
    	return ret;
    
    	
    }
    
    public static CvMat calculate3dPoint(CvMat p1, CvMat c1, CvMat p2, CvMat c2)
    {
    	CvMat X = CvMat.create(3, 1,CV_32F);
    	CvMat A = CvMat.create(4, 3,CV_32F);
    	CvMat B = CvMat.create(4, 1,CV_32F);
    	
    	A.put(0 , 0 , p1.get(0,0) * c1.get(2,0) - c1.get(0,0) );
    	A.put(0 , 1 , p1.get(0,0) * c1.get(2,1) - c1.get(0,1) );
    	A.put(0 , 2 , p1.get(0,0) * c1.get(2,2) - c1.get(0,2) );
    	A.put(1 , 0 , p1.get(1,0) * c1.get(2,0) - c1.get(1,0) );
    	A.put(1 , 1 , p1.get(1,0) * c1.get(2,1) - c1.get(1,1) );
    	A.put(1 , 2 , p1.get(1,0) * c1.get(2,2) - c1.get(1,2) );
    	A.put(2 , 0 , p2.get(0,0) * c2.get(2,0) - c2.get(0,0) );
    	A.put(2 , 1 , p2.get(0,0) * c2.get(2,1) - c2.get(0,1) );
    	A.put(2 , 2 , p2.get(0,0) * c2.get(2,2) - c2.get(0,2) );
    	A.put(3 , 0 , p2.get(1,0) * c2.get(2,0) - c2.get(1,0) );
    	A.put(3 , 1 , p2.get(1,0) * c2.get(2,1) - c2.get(1,1) );
    	A.put(3 , 2 , p2.get(1,0) * c2.get(2,2) - c2.get(1,2) );
    	
    	B.put(0 , 0 , p1.get(0,0) * c1.get(2,3) - c1.get(0,3) );
    	B.put(1 , 0 , p1.get(1,0) * c1.get(2,3) - c1.get(1,3) );
    	B.put(2 , 0 , p2.get(0,0) * c2.get(2,3) - c2.get(0,3) );
    	B.put(3 , 0 , p2.get(1,0) * c2.get(2,3) - c2.get(1,3) );
    	
    	cvSolve(A, B, X,CV_SVD);
    	
    	return X;    
    }
    
    public static CvMat calculate3dPointIt(CvMat p1, CvMat c1, CvMat p2, CvMat c2)
    {
    	CvMat X = CvMat.create(4, 1,CV_32F);
    	CvMat A = CvMat.create(4, 3,CV_32F);
    	CvMat B = CvMat.create(4, 1,CV_32F);
    	CvPoint3D32f ret = new CvPoint3D32f();
    	
    	double wi=1;
    	double wi1=1;
    	
    	for (int i=0;i<10;i++)
    	{
    		CvMat X_ = calculate3dPoint(p1,c1,p2,c2);
    	    X.put(0,0,X_.get(0,0));
    	    X.put(1,0,X_.get(1,0));
    	    X.put(2,0,X_.get(2,0));
    	    X.put(3,0,1);
    	    CvMat help1 = CvMat.create(1, 4,CV_32F);
    	    help1.put(0,0,c1.get(2,0));
    	    help1.put(0,1,c1.get(2,1));
    	    help1.put(0,2,c1.get(2,2));
    	    help1.put(0,3,c1.get(2,3));
    	    CvMat help2 = CvMat.create(1, 1, CV_32F);
    	    cvMatMul(help1, X, help2);
    	    
            double p2x = help2.get(0, 0);
    	    help1.put(0,0,c2.get(2,0));
    	    help1.put(0,1,c2.get(2,1));
    	    help1.put(0,2,c2.get(2,2));
    	    help1.put(0,3,c2.get(2,3));
    	    cvMatMul(help1, X, help2);
            double p2x1 = help2.get(0, 0);
            
            wi = p2x;
            wi1 = p2x1;
            
            
    		A.put(0 , 0 , p1.get(0,0) * c1.get(2,0) - c1.get(0,0) /wi);
    		A.put(0 , 1 , p1.get(0,0) * c1.get(2,1) - c1.get(0,1) /wi);
    		A.put(0 , 2 , p1.get(0,0) * c1.get(2,2) - c1.get(0,2) /wi);
    		A.put(1 , 0 , p1.get(1,0) * c1.get(2,0) - c1.get(1,0) /wi);
    		A.put(1 , 1 , p1.get(1,0) * c1.get(2,1) - c1.get(1,1) /wi);
    		A.put(1 , 2 , p1.get(1,0) * c1.get(2,2) - c1.get(1,2) /wi);
    		A.put(2 , 0 , p2.get(0,0) * c2.get(2,0) - c2.get(0,0) /wi1);
    		A.put(2 , 1 , p2.get(0,0) * c2.get(2,1) - c2.get(0,1) /wi1);
    		A.put(2 , 2 , p2.get(0,0) * c2.get(2,2) - c2.get(0,2) /wi1);
    		A.put(3 , 0 , p2.get(1,0) * c2.get(2,0) - c2.get(1,0) /wi1);
    		A.put(3 , 1 , p2.get(1,0) * c2.get(2,1) - c2.get(1,1) /wi1);
    		A.put(3 , 2 , p2.get(1,0) * c2.get(2,2) - c2.get(1,2) /wi1);
    	
    		B.put(0 , 0 , p1.get(0,0) * c1.get(2,3) - c1.get(0,3) /wi);
    		B.put(1 , 0 , p1.get(1,0) * c1.get(2,3) - c1.get(1,3) /wi);
    		B.put(2 , 0 , p2.get(0,0) * c2.get(2,3) - c2.get(0,3) /wi1);
    		B.put(3 , 0 , p2.get(1,0) * c2.get(2,3) - c2.get(1,3) /wi1);
    	
    		cvSolve(A, B, X_,CV_SVD);
    	}
    	return X;
    }
    
    public static CvMat geCamMatInv()
    {
    	CvMat ret = CvMat.create(3, 3);
    	CvMat camMat = CvMat.create(3, 3);
    	ret.put(0,0,xscale/2);
    	ret.put(0,1,0.);
    	ret.put(0,2, xscale/2);
    	ret.put(1,0,0.);
    	ret.put(1,1,yscale/2);
    	ret.put(1,2, yscale/2);
    	ret.put(2,0, 0.);
    	ret.put(2,1, 0.);
    	ret.put(2,2, 1.);
    	cvInvert(ret, camMat);
    	return camMat;
    }
    public static CvMat geCamMat()
    {
    	CvMat ret = CvMat.create(3, 3,CV_32F);
    	ret.put(0,0,100);
    	ret.put(0,1,0.);
    	ret.put(0,2, 75);
    	ret.put(1,0,0.);
    	ret.put(1,1,100 );
    	ret.put(1,2, 100);
    	ret.put(2,0, 0.);
    	ret.put(2,1, 0.);
    	ret.put(2,2, 1.);
    	return ret;
    }
    
    public static CvMat buildMap(KeyPoint p1, CvMat c1, KeyPoint p2, CvMat c2, DMatch matches)
    {
    	int size_x = 1000;
    	int size_z = 1000;
    	CvMat KInv = geCamMatInv();
    	//System.out.println(KInv.toString());
    	
    	CvMat point1 = CvMat.create(3, 1);
    	CvMat point2 = CvMat.create(3, 1);
    	CvMat point1_inv = CvMat.create(3, 1);
    	CvMat point2_inv = CvMat.create(3, 1);
    	CvMat map = CvMat.create(size_x, size_z);
    	ArrayList<CvMat> points = new ArrayList<CvMat>();
    	for (int i =0; i < matches.capacity() && i <p2.capacity();i++)
    	{
    		point1.put(0,0,p1.position(matches.position(i).queryIdx()).pt_x());
    		point1.put(0,1,p1.position(matches.position(i).queryIdx()).pt_y());
    		point1.put(0,2,1);
    		point2.put(0,0,p2.position(matches.position(i).trainIdx()).pt_x());
    		point2.put(0,1,p2.position(matches.position(i).trainIdx()).pt_y());
    		point2.put(0,2,1);    		    		
    		cvMatMul(KInv, point1, point1_inv);
    		cvMatMul(KInv, point2, point2_inv);
    		CvMat temp = calculate3dPointIt( point1_inv, c1, point2_inv, c2);
			int x = (int) Math.floor(temp.get(0,0)) + size_x/2;
			int z = (int) Math.floor(temp.get(2,0)) + size_z/2;
    		if(x < size_x && z < size_z && x > 0 && z > 0)
    		{
    			points.add(temp);
    			map.put(z,x,temp.get(1,0)+50);
    			//gas.input(temp);
    			//System.out.println(temp.get(0,0)+"\t"+temp.get(0,1)+"\t"+temp.get(0,2));
    			
    		}
    	}
    	gas.input(points);
    	matches.position(0);
    	return map;
    	
    }
    
    public static CvMat getEssentialMatrix(CvMat P1, CvMat P2)
    {
    	CvMat ret = CvMat.create(3, 3,P1.depth());
    	CvMat tx = CvMat.create(3, 3,P1.depth());
    	CvMat r = CvMat.create(3, 3,P1.depth());
    	tx.put(0,0,0);
    	tx.put(0,1,- P1.get(0,2) - P2.get(0,2) );
    	tx.put(0,2,  P1.get(0,1) + P2.get(0,1) );
    	tx.put(1,0, P1.get(0,2) + P2.get(0,2) );
    	tx.put(1,1,0);
    	tx.put(1,2,- P1.get(0,0) - P2.get(0,0) );
    	tx.put(2,0,- P1.get(0,1) - P2.get(0,1) );
    	tx.put(2,1, P1.get(0,0) + P2.get(0,0) );
    	tx.put(2,2,0);
    	
    	r.put(0,0,P1.get(0,0)+P2.get(0,0));
    	r.put(0,1,P1.get(0,1)+P2.get(0,1));
    	r.put(0,2,P1.get(0,2)+P2.get(0,2));
    	r.put(1,0,P1.get(1,0)+P2.get(1,0));
    	r.put(1,1,P1.get(1,1)+P2.get(1,1));
    	r.put(1,2,P1.get(1,2)+P2.get(1,2));
    	r.put(2,0,P1.get(2,0)+P2.get(2,0));
    	r.put(2,1,P1.get(2,1)+P2.get(2,1));
    	r.put(2,2,P1.get(2,2)+P2.get(2,2));
    	
    	cvMatMul(tx, r, ret);
    	
    	return ret;
    	
    }
    
    public static CvMat build3dMap(int im_1,int im_2)
    {
    	//System.out.println("Loading images...");
    	IplImage image1 = openImage("res/gnome-0-"+im_1+"-0.jpg");
    	IplImage image2 = openImage("res/gnome-0-"+im_2+"-0.jpg");
    	//System.out.println("resize images...");
    	image1 = resize(image1, xscale, yscale);
    	image2 = resize(image2, xscale, yscale);

    	//System.out.println("Finding Keypoint...");
    	KeyPoint v1 = findKeypoints(image1);
    	KeyPoint v2 = findKeypoints(image2);


    	//System.out.println("get camera matrices...");
   	    CvMat rot = getRotationMatrix(0);
   	    CvMat tran = getTransitionMatrix(im_1,0,0);
   	    CvMat P1 = getCameraMatrix(rot, tran);
   	    
   	    rot = getRotationMatrix(0);
	    tran = getTransitionMatrix(im_2,0,0);
   	    CvMat P2 = getCameraMatrix(rot, tran);
   	    
   	    CvMat e = getEssentialMatrix(P1, P2);
   	    
    	//System.out.println("Matching Keypoint...");
    	DMatch good_matches = findMatchingFeature(image1, image2, v1, v2,e);
   	    

    	//System.out.println("Building Map...");
   	    CvMat map = buildMap(v1,P1,v2,P2,good_matches);
   	    image1=null;
   	    image2=null;
   	    return map;
    }
    
    	
}